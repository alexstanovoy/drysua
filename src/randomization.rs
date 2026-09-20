//! Annealed domain randomization: schedule, per-generation draws and their
//! on-disk snapshots.
//!
//! Every draw is a pure function of `(run seed, generation index, schedule)`.
//! Sampling uses only integer arithmetic and the crate's splitmix-based
//! [`PpoRng`], so a resumed run recomputes the exact same generations.

use std::path::Path;

use bota_proto::ModifierSpec;

use crate::PpoError;
use crate::PpoRng;
use crate::model::{FNV_OFFSET, fnv1a_extend};

/// Schema tag of one generation snapshot file.
pub const RANDOMIZATION_SCHEMA: &str = "drysua-domain-randomization/v2";
/// Directory of generation snapshots inside a checkpoint directory.
pub const RANDOMIZATION_DIRECTORY: &str = "domain-randomization";
/// Nominal value of a basis-point rate, one hundred percent.
pub const NOMINAL_BP: i32 = 10_000;
/// One standard deviation is this fraction of a variable's range.
const SIGMA_DIVISOR: i32 = 3;
/// Independent uniform draws summed for the bounded normal approximation.
const NORMAL_DRAWS: usize = 12;
/// Width of one uniform draw, as a power of two.
const NORMAL_DRAW_BITS: u32 = 32;
/// Domain separating generation draws from other derived streams.
const GENERATION_DOMAIN: u64 = 0x6765_6e65_7261_7465;
/// Domain separating per-game arena seeds from other derived streams.
pub(crate) const ARENA_DOMAIN: u64 = 0x6172_656e_615f_7365;
/// Domain separating per-game opponent streams.
pub(crate) const OPPONENT_DOMAIN: u64 = 0x6f70_706f_6e65_6e74;
/// Largest accepted generation snapshot file, one small JSON line.
const MAX_SNAPSHOT_BYTES: u64 = 4 * 1024;

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
/// `max_hp` alone is two-sided; the one-sided variables are truncated at
/// nominal, which makes them half-normal over their span.
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

/// One update-budget annealing schedule.
///
/// `scale_bp(u)` is `max(0, 1 - sqrt(u / (updates - zero_updates)))` in basis
/// points; the last `zero_updates` updates always scale to zero. A run whose
/// zero window covers every update is all-zero, not undefined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnnealSchedule {
    /// Total updates in the run.
    pub updates: u64,
    /// Final updates played with no modifiers.
    pub zero_updates: u64,
}

impl AnnealSchedule {
    /// The first update whose scale is zero.
    pub const fn zero_from_update(&self) -> u64 {
        self.updates.saturating_sub(self.zero_updates)
    }

    /// The scale at one update, in basis points of full variance.
    pub fn scale_bp(&self, update: u64) -> i32 {
        let span = self.zero_from_update();
        if span == 0 || update >= span {
            return 0;
        }
        let numerator = u128::from(update) * (NOMINAL_BP as u128) * (NOMINAL_BP as u128);
        let root = isqrt_u128(numerator / u128::from(span));
        let root = i32::try_from(root).unwrap_or(NOMINAL_BP);
        (NOMINAL_BP - root).max(0)
    }
}

/// One generation: its games, the update it started in, and its draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GenerationDraw {
    /// Global generation index, counted in generations of `games_per_generation`.
    pub generation: u64,
    /// First global game index of the generation.
    pub start_game: u64,
    /// One past the last global game index of the generation.
    pub end_game: u64,
    /// Update the generation started in; its scale is the update's scale.
    pub start_update: u64,
    /// Annealing scale at `start_update`, in basis points.
    pub scale_bp: i32,
    /// Games of the generation that actually carry the draw: the zero window
    /// truncates a generation that crosses into it, and a fully truncated
    /// generation carries none.
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
/// `games_per_generation` and `games_per_update` map the generation back to
/// its first game and update so the scale is the one the generation started
/// under. A zero scale draws nothing and returns a nominal spec; a generation
/// crossing into the zero window records how many of its games still carry
/// the draw.
pub fn draw_generation(
    seed: u64,
    generation: u64,
    games_per_generation: u64,
    games_per_update: u64,
    schedule: AnnealSchedule,
) -> Result<GenerationDraw, PpoError> {
    assert!(games_per_generation > 0);
    assert!(games_per_update > 0);
    let start_game = generation
        .checked_mul(games_per_generation)
        .ok_or(PpoError::CounterOverflow)?;
    let end_game = start_game
        .checked_add(games_per_generation)
        .ok_or(PpoError::CounterOverflow)?;
    let start_update = start_game / games_per_update;
    let scale_bp = schedule.scale_bp(start_update);
    let zero_from_game = schedule
        .zero_from_update()
        .checked_mul(games_per_update)
        .ok_or(PpoError::CounterOverflow)?;
    let applied_games = end_game.min(zero_from_game).saturating_sub(start_game);
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

/// One bounded normal draw in spec units, from twelve uniform words.
///
/// The sum of twelve independent uniforms has mean six and variance one; the
/// result is truncated at six standard deviations and rounded half up. Only
/// integer arithmetic runs, so the value is identical on every platform.
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

/// Reads one snapshot, refusing anything past the small-file bound.
fn read_snapshot(path: &Path) -> Result<String, PpoError> {
    let metadata = std::fs::metadata(path).map_err(|_| {
        PpoError::InvalidConfig("domain randomization snapshot is missing on resume")
    })?;
    if metadata.len() > MAX_SNAPSHOT_BYTES {
        return Err(PpoError::InvalidConfig(
            "domain randomization snapshot is oversized",
        ));
    }
    std::fs::read_to_string(path)
        .map_err(|error| PpoError::Model(format!("randomization snapshot read: {error}")))
}

/// Writes one generation snapshot, verifying an existing file byte for byte.
///
/// A mismatch means the recomputed generation disagrees with what a previous
/// run recorded; the run is stopped rather than continued under different
/// world modifiers. File contents are synced before a durable rename: Unix
/// syncs the parent directory and Windows uses `MOVEFILE_WRITE_THROUGH`,
/// matching the checkpoint path.
pub fn write_generation_snapshot(directory: &Path, draw: &GenerationDraw) -> Result<(), PpoError> {
    use std::io::Write;

    std::fs::create_dir_all(directory)
        .map_err(|error| PpoError::Model(format!("randomization directory: {error}")))?;
    let path = generation_path(directory, draw.generation);
    let text = generation_json(draw);
    if path.exists() {
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
    crate::checkpoint::durable_rename(&temporary, &path)
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
    games_per_update: u64,
    schedule: AnnealSchedule,
    completed_games: u64,
) -> Result<u64, PpoError> {
    let mut generation = 0u64;
    loop {
        let start_game = generation
            .checked_mul(games_per_generation)
            .ok_or(PpoError::CounterOverflow)?;
        if start_game >= completed_games {
            return Ok(generation);
        }
        let draw = draw_generation(
            seed,
            generation,
            games_per_generation,
            games_per_update,
            schedule,
        )?;
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
