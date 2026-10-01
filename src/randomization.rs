//! Annealed domain randomization: schedule, per-generation draws and their
//! on-disk snapshots.
//!
//! Every draw is a pure function of the run seed, the generation index and the
//! scale at its start update. Sampling uses only integer arithmetic and
//! [`PpoRng`], so a resumed run recomputes the same generations on any platform.

use std::path::Path;

use bota_proto::ModifierSpec;

use crate::PpoError;
use crate::PpoRng;
use crate::durability::RegularFileError;
use crate::model::{FNV_OFFSET, fnv1a_extend};

/// Schema tag of one generation snapshot file.
pub const RANDOMIZATION_SCHEMA: &str = "drysua-domain-randomization/v2";
/// Directory of generation snapshots inside a checkpoint directory.
#[cfg(feature = "builtin")]
pub const RANDOMIZATION_DIRECTORY: &str = "domain-randomization";
/// Nominal value of a basis-point rate, one hundred percent.
pub const NOMINAL_BP: i32 = 10_000;
/// Full-scale sigma is a variable's `sigma_range` divided by this.
const SIGMA_DIVISOR: i32 = 3;
/// Independent uniform draws summed for the bounded normal approximation.
const NORMAL_DRAWS: usize = 12;
/// Width of one uniform draw, as a power of two.
const NORMAL_DRAW_BITS: u32 = 32;
/// Domain separating generation draws from other derived streams.
const GENERATION_DOMAIN: u64 = 0x6765_6e65_7261_7465;
/// Domain separating per-game arena seeds from other derived streams.
#[cfg(feature = "builtin")]
pub(crate) const ARENA_DOMAIN: u64 = 0x6172_656e_615f_7365;
/// Domain separating per-game opponent streams.
#[cfg(feature = "builtin")]
pub(crate) const OPPONENT_DOMAIN: u64 = 0x6f70_706f_6e65_6e74;
/// Largest accepted generation snapshot file, one small JSON line.
pub(crate) const MAX_SNAPSHOT_BYTES: u64 = 4096;

/// One randomized variable: its bounds in spec units and the field it writes.
#[derive(Clone, Copy, Debug)]
pub struct RandomizationVariable {
    /// Stable name used in snapshots and logs.
    pub name: &'static str,
    /// Spec value at a zero delta.
    pub nominal: i32,
    /// Smallest accepted delta from nominal.
    pub lower: i32,
    /// Largest accepted delta from nominal.
    pub upper: i32,
    /// Three-sigma span at full scale.
    pub sigma_range: i32,
    set: fn(&mut ModifierSpec, i32),
}

impl RandomizationVariable {
    /// Writes the absolute spec value for one delta.
    pub fn apply(&self, spec: &mut ModifierSpec, delta: i32) {
        (self.set)(spec, self.nominal + delta);
    }
}

fn set_max_hp(spec: &mut ModifierSpec, value: i32) {
    spec.max_hp = value;
}

fn set_gold_income(spec: &mut ModifierSpec, value: i32) {
    spec.gold_income = value;
}

fn set_max_mana(spec: &mut ModifierSpec, value: i32) {
    spec.max_mana = value;
}

fn set_physical_damage(spec: &mut ModifierSpec, value: i32) {
    spec.physical_damage = value;
}

fn set_magic_damage(spec: &mut ModifierSpec, value: i32) {
    spec.magic_damage = value;
}

fn set_pure_damage(spec: &mut ModifierSpec, value: i32) {
    spec.pure_damage = value;
}

fn set_magic_resist(spec: &mut ModifierSpec, value: i32) {
    spec.magic_resist = value;
}

fn set_status_resist(spec: &mut ModifierSpec, value: i32) {
    spec.status_resist = value;
}

fn set_move_speed(spec: &mut ModifierSpec, value: i32) {
    spec.move_speed = value;
}

fn set_mana_cost_rate(spec: &mut ModifierSpec, value: i32) {
    spec.mana_cost_rate = value;
}

fn set_cooldown_rate(spec: &mut ModifierSpec, value: i32) {
    spec.cooldown_rate = value;
}

/// The eleven randomized variables, in snapshot order.
///
/// `max_hp` alone is two-sided; the one-sided variables clamp at nominal, so
/// half their draws land exactly on nominal and the rest are half-normal.
pub const VARIABLES: [RandomizationVariable; 11] = [
    RandomizationVariable {
        name: "max_hp",
        nominal: NOMINAL_BP,
        lower: -4_000,
        upper: 10_000,
        sigma_range: 10_000,
        set: set_max_hp,
    },
    RandomizationVariable {
        name: "gold_income",
        nominal: NOMINAL_BP,
        lower: 0,
        upper: 10_000,
        sigma_range: 10_000,
        set: set_gold_income,
    },
    RandomizationVariable {
        name: "max_mana",
        nominal: NOMINAL_BP,
        lower: 0,
        upper: 10_000,
        sigma_range: 10_000,
        set: set_max_mana,
    },
    RandomizationVariable {
        name: "physical_damage",
        nominal: NOMINAL_BP,
        lower: 0,
        upper: 10_000,
        sigma_range: 10_000,
        set: set_physical_damage,
    },
    RandomizationVariable {
        name: "magic_damage",
        nominal: NOMINAL_BP,
        lower: 0,
        upper: 10_000,
        sigma_range: 10_000,
        set: set_magic_damage,
    },
    RandomizationVariable {
        name: "pure_damage",
        nominal: NOMINAL_BP,
        lower: 0,
        upper: 10_000,
        sigma_range: 10_000,
        set: set_pure_damage,
    },
    RandomizationVariable {
        name: "magic_resist",
        nominal: 0,
        lower: 0,
        upper: 4_000,
        sigma_range: 4_000,
        set: set_magic_resist,
    },
    RandomizationVariable {
        name: "status_resist",
        nominal: 0,
        lower: 0,
        upper: 5_000,
        sigma_range: 5_000,
        set: set_status_resist,
    },
    RandomizationVariable {
        name: "move_speed",
        nominal: NOMINAL_BP,
        lower: 0,
        upper: 3_000,
        sigma_range: 3_000,
        set: set_move_speed,
    },
    RandomizationVariable {
        name: "mana_cost_rate",
        nominal: NOMINAL_BP,
        lower: -5_000,
        upper: 0,
        sigma_range: 5_000,
        set: set_mana_cost_rate,
    },
    RandomizationVariable {
        name: "cooldown_rate",
        nominal: NOMINAL_BP,
        lower: -5_000,
        upper: 0,
        sigma_range: 5_000,
        set: set_cooldown_rate,
    },
];

/// Annealing schedule over a run's updates.
///
/// `scale_bp(u)` interpolates from `scale.start_bp` towards `scale.end_bp` by
/// `sqrt(u / (updates - zero_updates))`; the last `zero_updates` updates always
/// scale to zero, and a zero window covering every update makes the run all-zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnnealSchedule {
    /// Total updates in the run.
    pub updates: u64,
    /// Final updates played with no modifiers.
    pub zero_updates: u64,
    /// Scale ramp endpoints; the default is [`AnnealScale::FULL`].
    pub scale: AnnealScale,
}

impl AnnealSchedule {
    /// First update of the zero-scale tail.
    pub const fn zero_from_update(&self) -> u64 {
        self.updates.saturating_sub(self.zero_updates)
    }

    /// Scale at one update, in basis points of full-scale sigma. The zero tail is
    /// zero whatever the ramp endpoints are.
    pub fn scale_bp(&self, update: u64) -> i32 {
        let span = self.zero_from_update();
        if span == 0 || update >= span {
            return 0;
        }
        let numerator = u128::from(update) * (NOMINAL_BP as u128) * (NOMINAL_BP as u128);
        let root = isqrt_u128(numerator / u128::from(span));
        let root = i32::try_from(root).unwrap_or(NOMINAL_BP);
        self.scale.apply(root)
    }
}

/// Environment scale ramp endpoints, in basis points of full-scale sigma
/// (`NOMINAL_BP` is one hundred percent, `MAX_BP` ten times that).
///
/// Variable ranges are constants, so a larger scale widens the sampling sigmas
/// while every sampled delta still clamps to its variable's bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnnealScale {
    /// Scale at the first update of the ramp, in basis points.
    pub start_bp: i32,
    /// Scale approached at the zero-tail boundary, in basis points.
    pub end_bp: i32,
}

impl AnnealScale {
    /// Nominal scale at the start, ramping to zero at the zero tail.
    pub const FULL: Self = Self {
        start_bp: NOMINAL_BP,
        end_bp: 0,
    };
    /// Largest admitted scale, ten times nominal.
    pub const MAX_BP: i32 = 10 * NOMINAL_BP;

    /// Rejects an endpoint outside `0..=MAX_BP`.
    pub fn validate(self) -> Result<Self, PpoError> {
        if !(0..=Self::MAX_BP).contains(&self.start_bp)
            || !(0..=Self::MAX_BP).contains(&self.end_bp)
        {
            return Err(PpoError::InvalidConfig(
                "environment scale must be within zero and ten times full variance",
            ));
        }
        Ok(self)
    }

    /// Interpolates the ramp for one normalized root in `0..=NOMINAL_BP`.
    ///
    /// Integer arithmetic only, so the value is identical on every platform.
    pub const fn apply(self, root: i32) -> i32 {
        let numerator = self.start_bp as i64 * (NOMINAL_BP as i64 - root as i64)
            + self.end_bp as i64 * root as i64;
        (numerator / NOMINAL_BP as i64) as i32
    }

    /// Scope flags with a leading space; recorded only for a non-default ramp.
    pub fn scope_suffix(self) -> String {
        format!(
            " --environment-scale-start {} --environment-scale-end {}",
            decimal_text(self.start_bp),
            decimal_text(self.end_bp)
        )
    }
}

/// Endpoint as the command-line decimal, where `1` is `NOMINAL_BP`.
fn decimal_text(bp: i32) -> crate::EnvironmentDecimal {
    crate::EnvironmentDecimal::from_units(bp as u64 * 100)
}

/// Parses an endpoint into basis points; rejects values finer than one basis
/// point or above `MAX_BP`.
fn canonical_bp(value: &str) -> Option<i32> {
    let units = value.parse::<crate::EnvironmentDecimal>().ok()?.units();
    if !units.is_multiple_of(100) || units > (AnnealScale::MAX_BP as u64) * 100 {
        return None;
    }
    i32::try_from(units / 100).ok()
}

/// Splits the trailing scale flags, if any, off a scope command line.
///
/// Only the exact rendering of a non-default ramp is split, so a partial,
/// malformed or default-valued `--environment-scale-*` token stays in the
/// returned prefix, where the scope validator rejects it.
pub(crate) fn split_scale_scope(command: &str) -> (&str, AnnealScale) {
    let tokens: Vec<&str> = command.split(' ').collect();
    if tokens.len() < 4
        || tokens[tokens.len() - 4] != "--environment-scale-start"
        || tokens[tokens.len() - 2] != "--environment-scale-end"
    {
        return (command, AnnealScale::FULL);
    }
    let (Some(start_bp), Some(end_bp)) = (
        canonical_bp(tokens[tokens.len() - 3]),
        canonical_bp(tokens[tokens.len() - 1]),
    ) else {
        return (command, AnnealScale::FULL);
    };
    let scale = AnnealScale { start_bp, end_bp };
    if scale == AnnealScale::FULL {
        // The default ramp is never recorded, so default tokens are not canonical.
        return (command, AnnealScale::FULL);
    }
    match command.strip_suffix(&scale.scope_suffix()) {
        Some(prefix) => (prefix, scale),
        None => (command, AnnealScale::FULL),
    }
}

/// One generation: its games, the update it started in, and its draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GenerationDraw {
    /// Global generation index.
    pub generation: u64,
    /// First global game index of the generation.
    pub start_game: u64,
    /// One past the last global game index of the generation; for adaptive
    /// generations, the end of the run.
    pub end_game: u64,
    /// Update the generation started in; its scale is the update's scale.
    pub start_update: u64,
    /// Annealing scale at `start_update`, in basis points.
    pub scale_bp: i32,
    /// Games of the generation that carry the draw: the zero window truncates
    /// a generation that crosses into it. For adaptive generations this is an
    /// upper bound, since the actual end is not known when drawing.
    pub applied_games: u64,
    /// Per-variable deltas from nominal, in spec units.
    pub deltas: [i32; VARIABLES.len()],
    /// The modifier every hero carries for the generation.
    pub spec: ModifierSpec,
}

impl GenerationDraw {
    /// Whether any game of the generation carries the draw.
    pub const fn applies(&self) -> bool {
        self.applied_games > 0 && self.scale_bp > 0
    }
}

/// Derives one generation's modifier from the run seed and the schedule.
///
/// A "game" of the snapshot format is one update's collection, so
/// `games_per_generation` maps the generation back to its first update and
/// the scale is the one the generation started under. A zero scale draws
/// nothing and returns a nominal spec; a generation crossing into the zero
/// window records how many of its updates still carry the draw.
pub fn draw_generation(
    seed: u64,
    generation: u64,
    games_per_generation: u64,
    schedule: AnnealSchedule,
) -> Result<GenerationDraw, PpoError> {
    assert!(games_per_generation > 0);
    let start_game = generation
        .checked_mul(games_per_generation)
        .ok_or(PpoError::CounterOverflow)?;
    let end_game = start_game
        .checked_add(games_per_generation)
        .ok_or(PpoError::CounterOverflow)?;
    let start_update = start_game;
    let scale_bp = schedule.scale_bp(start_update);
    let applied_games = end_game
        .min(schedule.zero_from_update())
        .saturating_sub(start_game);
    let (deltas, spec) = draw_modifiers(seed, generation, scale_bp)?;
    Ok(GenerationDraw {
        generation,
        start_game,
        end_game,
        start_update,
        scale_bp,
        applied_games,
        deltas,
        spec,
    })
}

/// Draws a generation starting at the adaptive controller's `start_update`; the
/// RNG stream is still keyed by generation index. End and applied-game counts are
/// run-wide bounds, not the generation's actual duration.
pub(crate) fn draw_generation_at_start(
    seed: u64,
    generation: u64,
    start_update: u64,
    schedule: AnnealSchedule,
) -> Result<GenerationDraw, PpoError> {
    if start_update >= schedule.updates {
        return Err(PpoError::InvalidConfig(
            "adaptive randomization draw requires a start before total updates",
        ));
    }
    let start_game = start_update;
    let end_game = schedule.updates;
    let applied_games = schedule.zero_from_update().saturating_sub(start_update);
    let scale_bp = schedule.scale_bp(start_update);
    let (deltas, spec) = draw_modifiers(seed, generation, scale_bp)?;
    debug_assert!(start_game < end_game);
    debug_assert!(applied_games <= end_game - start_game);
    Ok(GenerationDraw {
        generation,
        start_game,
        end_game,
        start_update,
        scale_bp,
        applied_games,
        deltas,
        spec,
    })
}

fn draw_modifiers(
    seed: u64,
    generation: u64,
    scale_bp: i32,
) -> Result<([i32; VARIABLES.len()], ModifierSpec), PpoError> {
    assert!((0..=AnnealScale::MAX_BP).contains(&scale_bp));
    let mut deltas = [0i32; VARIABLES.len()];
    let mut spec = ModifierSpec::NOMINAL;
    if scale_bp > 0 {
        let stream_seed = derive_training_seed(seed, generation, GENERATION_DOMAIN);
        let mut rng = PpoRng::new(stream_seed);
        for (index, variable) in VARIABLES.iter().enumerate() {
            let sigma = i64::from(scale_bp) * i64::from(variable.sigma_range)
                / (SIGMA_DIVISOR as i64 * i64::from(NOMINAL_BP));
            let sampled = normal_delta(&mut rng, sigma as i32)?;
            let delta = sampled.clamp(variable.lower, variable.upper);
            deltas[index] = delta;
            variable.apply(&mut spec, delta);
        }
    }
    assert!(spec.is_bounded(), "a draw stays inside the spec bounds");
    Ok((deltas, spec))
}

/// One bounded normal draw in spec units, from twelve uniform words.
///
/// The sum of twelve independent uniforms has mean six and variance one, so the
/// centered result is bounded by six standard deviations; it is rounded half up.
/// Only integer arithmetic runs, so the value is identical on every platform.
fn normal_delta(rng: &mut PpoRng, sigma: i32) -> Result<i32, PpoError> {
    if sigma == 0 {
        return Ok(0);
    }
    let mut sum = 0i64;
    for _ in 0..NORMAL_DRAWS {
        sum += i64::try_from(rng.below(1u64 << NORMAL_DRAW_BITS)?)
            .map_err(|_| PpoError::CounterOverflow)?;
    }
    let centered = sum - 6 * (1i64 << NORMAL_DRAW_BITS);
    let product = i64::from(sigma)
        .checked_mul(centered)
        .ok_or(PpoError::CounterOverflow)?;
    let rounded = (product + (1i64 << (NORMAL_DRAW_BITS - 1))) >> NORMAL_DRAW_BITS;
    i32::try_from(rounded).map_err(|_| PpoError::CounterOverflow)
}

/// A stable stream seed derived from a base, an index and a domain.
pub(crate) const fn derive_training_seed(base: u64, stream: u64, domain: u64) -> u64 {
    let mut value = base ^ stream.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ domain;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Integer square root, rounded down.
pub(crate) fn isqrt_u128(value: u128) -> u128 {
    if value <= 1 {
        return value;
    }
    let mut root = 0u128;
    let mut bit = 1u128 << 126;
    let mut rest = value;
    while bit > rest {
        bit >>= 2;
    }
    while bit != 0 {
        let grown = root + bit;
        if rest >= grown {
            rest -= grown;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }
        bit >>= 2;
    }
    root
}

/// Canonical one-line JSON for one generation snapshot.
///
/// Field order and integer formatting are fixed, so two renders of the same
/// draw are byte-identical and a stored snapshot can be compared without a
/// parser.
pub fn generation_json(draw: &GenerationDraw) -> String {
    let mut body = String::with_capacity(512);
    body.push_str("{\"schema\":\"");
    body.push_str(RANDOMIZATION_SCHEMA);
    body.push_str("\",\"generation\":");
    body.push_str(&draw.generation.to_string());
    body.push_str(",\"start_game\":");
    body.push_str(&draw.start_game.to_string());
    body.push_str(",\"end_game\":");
    body.push_str(&draw.end_game.to_string());
    body.push_str(",\"start_update\":");
    body.push_str(&draw.start_update.to_string());
    body.push_str(",\"scale_bp\":");
    body.push_str(&draw.scale_bp.to_string());
    body.push_str(",\"applied_games\":");
    body.push_str(&draw.applied_games.to_string());
    body.push_str(",\"deltas\":{");
    for (index, variable) in VARIABLES.iter().enumerate() {
        if index > 0 {
            body.push(',');
        }
        body.push('"');
        body.push_str(variable.name);
        body.push_str("\":");
        body.push_str(&draw.deltas[index].to_string());
    }
    body.push_str("},\"spec\":{");
    for (index, variable) in VARIABLES.iter().enumerate() {
        if index > 0 {
            body.push(',');
        }
        body.push('"');
        body.push_str(variable.name);
        body.push_str("\":");
        body.push_str(&(variable.nominal + draw.deltas[index]).to_string());
    }
    body.push('}');
    let hash = fnv1a_extend(FNV_OFFSET, body.as_bytes());
    body.push_str(",\"hash\":\"");
    body.push_str(&format!("{hash:016x}"));
    body.push_str("\"}\n");
    body
}

/// Reads one snapshot, refusing a symlink or anything past the small-file bound.
fn read_snapshot(path: &Path) -> Result<String, PpoError> {
    let bytes = crate::durability::read_regular_file(path, MAX_SNAPSHOT_BYTES).map_err(
        |error| match error {
            RegularFileError::Missing => {
                PpoError::InvalidConfig("domain randomization snapshot is missing on resume")
            }
            RegularFileError::NotRegular => {
                PpoError::InvalidConfig("domain randomization snapshot must be a regular file")
            }
            RegularFileError::Oversized => {
                PpoError::InvalidConfig("domain randomization snapshot is oversized")
            }
            RegularFileError::Changed => {
                PpoError::InvalidConfig("domain randomization snapshot length changed")
            }
            RegularFileError::Io(error) => {
                PpoError::Model(format!("randomization snapshot read: {error}"))
            }
        },
    )?;
    String::from_utf8(bytes)
        .map_err(|_| PpoError::InvalidConfig("domain randomization snapshot mismatch"))
}

/// Whether the snapshot directory exists, refusing a symlink or non-directory there.
fn snapshot_directory_exists(directory: &Path) -> Result<bool, PpoError> {
    crate::durability::real_directory_exists(directory).map_err(|error| match error {
        RegularFileError::Io(error) => PpoError::Model(format!("randomization directory: {error}")),
        _ => PpoError::InvalidConfig("domain randomization directory must be a real directory"),
    })
}

/// Writes generation snapshots, verifying existing files byte for byte.
///
/// A mismatch means the recomputed generation disagrees with what a previous
/// run recorded, so the run stops rather than continue under different world
/// modifiers. Each new file is synced before its rename and the directory once
/// at the end, so a batch costs one directory sync.
pub fn write_generation_snapshots(
    directory: &Path,
    draws: &[GenerationDraw],
) -> Result<(), PpoError> {
    if !snapshot_directory_exists(directory)? {
        std::fs::create_dir_all(directory)
            .map_err(|error| PpoError::Model(format!("randomization directory: {error}")))?;
    }
    for draw in draws {
        write_generation_file(directory, draw)?;
    }
    crate::durability::sync_directory(directory)
        .map_err(|error| PpoError::Model(format!("randomization snapshot commit: {error}")))
}

fn write_generation_file(directory: &Path, draw: &GenerationDraw) -> Result<(), PpoError> {
    use std::io::Write;

    let path = generation_path(directory, draw.generation);
    let text = generation_json(draw);
    let present = crate::durability::entry_exists(&path)
        .map_err(|error| PpoError::Model(format!("randomization snapshot metadata: {error}")))?;
    if present {
        let stored = read_snapshot(&path)?;
        if stored != text {
            return Err(PpoError::InvalidConfig(
                "domain randomization snapshot mismatch",
            ));
        }
        return Ok(());
    }
    let temporary = path.with_extension("json.tmp");
    let mut file = std::fs::File::create(&temporary)
        .map_err(|error| PpoError::Model(format!("randomization snapshot create: {error}")))?;
    file.write_all(text.as_bytes())
        .map_err(|error| PpoError::Model(format!("randomization snapshot write: {error}")))?;
    file.sync_all()
        .map_err(|error| PpoError::Model(format!("randomization snapshot sync: {error}")))?;
    drop(file);
    crate::checkpoint::commit_rename(&temporary, &path)
        .map_err(|error| PpoError::Model(format!("randomization snapshot commit: {error}")))
}

/// Verifies every generation started before `completed_games` against disk.
///
/// Used on resume: the snapshot chain is recomputed from the seed and compared
/// to the files, so a resume under changed ranges, schedule or seed stops
/// before any game is played. Returns how many generations were verified.
pub fn verify_generation_snapshots(
    directory: &Path,
    seed: u64,
    games_per_generation: u64,
    schedule: AnnealSchedule,
    completed_games: u64,
) -> Result<u64, PpoError> {
    snapshot_directory_exists(directory)?;
    let mut generation = 0u64;
    loop {
        let start_game = generation
            .checked_mul(games_per_generation)
            .ok_or(PpoError::CounterOverflow)?;
        if start_game >= completed_games {
            return Ok(generation);
        }
        let draw = draw_generation(seed, generation, games_per_generation, schedule)?;
        let path = generation_path(directory, generation);
        let stored = read_snapshot(&path)?;
        if stored != generation_json(&draw) {
            return Err(PpoError::InvalidConfig(
                "domain randomization snapshot mismatch",
            ));
        }
        generation = generation.checked_add(1).ok_or(PpoError::CounterOverflow)?;
    }
}

fn generation_path(directory: &Path, generation: u64) -> std::path::PathBuf {
    directory.join(format!("generation-{generation:012}.json"))
}

#[cfg(test)]
#[path = "tests/randomization.rs"]
mod tests;
