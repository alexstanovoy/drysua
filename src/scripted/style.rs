//! Seeded play styles for the rule policies.
//!
//! A style is a handful of integer knobs a rule policy reads in place of constants. Every
//! knob's default reproduces the canonical policy exactly. A [`StyleSpec`] fixes or ranges
//! knobs, and one seed per game and seat draws a concrete style from it, so a styled opponent
//! plays differently from game to game while each game stays reproducible and replayable.

use crate::scripted::ScriptKind;

/// Knobs one spec may name, far above any policy's table.
const MAX_SPEC_KNOBS: usize = 32;
/// Longest accepted spec text.
const MAX_SPEC_BYTES: usize = 1_024;
/// Domain separating style draws from the other per-game streams.
pub(crate) const STYLE_DOMAIN: u64 = 0x7374_796c_655f_6472;

/// One tunable of a rule policy: its bounds, its canonical value and its styled range.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Knob {
    pub name: &'static str,
    pub default: i32,
    pub min: i32,
    pub max: i32,
    /// Range the `styled` preset draws from.
    pub styled: (i32, i32),
}

impl Knob {
    pub(crate) const fn new(
        name: &'static str,
        default: i32,
        (min, max): (i32, i32),
        styled: (i32, i32),
    ) -> Self {
        assert!(min <= default && default <= max);
        assert!(min <= styled.0 && styled.0 <= styled.1 && styled.1 <= max);
        Self {
            name,
            default,
            min,
            max,
            styled,
        }
    }
}

/// Knobs every rule policy has: how often it decides and how often it acts at random.
pub(crate) const NOISE_KNOBS: [Knob; 2] = [
    // Decisions per acted decision; the others keep the running order.
    Knob::new("period", 1, (1, 4), (1, 2)),
    // Chance in a thousand that a decision is a random legal tactical action.
    Knob::new("epsilon", 0, (0, 500), (0, 40)),
];

/// Knob values in table order: the shared noise knobs, then the policy's own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StyleValues {
    values: [i32; MAX_SPEC_KNOBS],
    len: usize,
}

impl StyleValues {
    /// Canonical values of `kind`: the unstyled policy.
    pub fn canonical(kind: ScriptKind) -> Self {
        let mut values = [0; MAX_SPEC_KNOBS];
        let table = knobs(kind);
        for (value, knob) in values.iter_mut().zip(table.clone()) {
            *value = knob.default;
        }
        Self {
            values,
            len: table.count(),
        }
    }

    /// The value of the knob `name`; an unknown name is a programmer error.
    pub(crate) fn get(&self, kind: ScriptKind, name: &str) -> i32 {
        let index = knobs(kind)
            .position(|knob| knob.name == name)
            .unwrap_or_else(|| panic!("{} has no style knob {name}", kind.label()));
        assert!(index < self.len);
        self.values[index]
    }

    /// `name=value` pairs in table order, for logs.
    pub fn describe(&self, kind: ScriptKind) -> String {
        knobs(kind)
            .zip(&self.values[..self.len])
            .map(|(knob, value)| format!("{}={value}", knob.name))
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Which values each knob of one policy may take in a game.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StyleSpec {
    kind: ScriptKind,
    ranges: [(i32, i32); MAX_SPEC_KNOBS],
    len: usize,
}

impl StyleSpec {
    /// Every knob at its canonical value.
    pub fn canonical(kind: ScriptKind) -> Self {
        Self::from_table(kind, |knob| (knob.default, knob.default))
    }

    /// Every knob over its styled range.
    pub fn styled(kind: ScriptKind) -> Self {
        Self::from_table(kind, |knob| knob.styled)
    }

    fn from_table(kind: ScriptKind, range: impl Fn(&Knob) -> (i32, i32)) -> Self {
        let mut ranges = [(0, 0); MAX_SPEC_KNOBS];
        let mut len = 0;
        for (slot, knob) in ranges.iter_mut().zip(knobs(kind)) {
            *slot = range(knob);
            len += 1;
        }
        Self { kind, ranges, len }
    }

    /// Parses `[styled,]knob=value|knob=low..high,...`; unnamed knobs stay canonical, or
    /// styled after a leading `styled`. An empty text is the canonical policy.
    pub fn parse(kind: ScriptKind, text: &str) -> Result<Self, String> {
        if text.len() > MAX_SPEC_BYTES {
            return Err(format!("style spec longer than {MAX_SPEC_BYTES} bytes"));
        }
        let mut parts = text.split(',').filter(|part| !part.is_empty()).peekable();
        let mut spec = if parts.peek() == Some(&"styled") {
            parts.next();
            Self::styled(kind)
        } else {
            Self::canonical(kind)
        };
        for part in parts {
            let (name, value) = part
                .split_once('=')
                .ok_or_else(|| format!("style part `{part}` is not knob=value"))?;
            let (index, knob) = knobs(kind)
                .enumerate()
                .find(|(_, knob)| knob.name == name)
                .ok_or_else(|| {
                    let names: Vec<_> = knobs(kind).map(|knob| knob.name).collect();
                    format!(
                        "{} has no style knob `{name}`; knobs: {}",
                        kind.label(),
                        names.join(", ")
                    )
                })?;
            let number = |text: &str| {
                text.parse::<i32>()
                    .map_err(|error| format!("style knob `{name}`: {error}"))
            };
            let (low, high) = match value.split_once("..") {
                Some((low, high)) => (number(low)?, number(high)?),
                None => (number(value)?, number(value)?),
            };
            if low > high || low < knob.min || high > knob.max {
                return Err(format!(
                    "style knob `{name}` must lie in {}..{} with low <= high, got {value}",
                    knob.min, knob.max
                ));
            }
            spec.ranges[index] = (low, high);
        }
        Ok(spec)
    }

    pub const fn kind(&self) -> ScriptKind {
        self.kind
    }

    /// Whether every knob is fixed at its canonical value.
    pub fn is_canonical(&self) -> bool {
        *self == Self::canonical(self.kind)
    }

    /// Draws one style; the same seed always draws the same values.
    pub fn draw(&self, seed: u64) -> StyleValues {
        let mut state = seed ^ STYLE_DOMAIN;
        let mut values = [0; MAX_SPEC_KNOBS];
        for (value, (low, high)) in values.iter_mut().zip(&self.ranges[..self.len]) {
            let span = u64::from(high.abs_diff(*low)) + 1;
            *value = low + (splitmix64(&mut state) % span) as i32;
        }
        StyleValues {
            values,
            len: self.len,
        }
    }
}

/// The shared noise knobs followed by the policy's own table.
fn knobs(kind: ScriptKind) -> impl Iterator<Item = &'static Knob> + Clone {
    let own: &'static [Knob] = match kind {
        ScriptKind::Teacher => &crate::teacher::STYLE_KNOBS,
        ScriptKind::HarassPush => &crate::scripted::harass_push::STYLE_KNOBS,
    };
    NOISE_KNOBS.iter().chain(own)
}

/// The style seed of one seat of the game played on `arena_seed`.
pub fn seat_seed(arena_seed: u64, seat: usize) -> u64 {
    let mut state = arena_seed ^ (seat as u64).wrapping_add(1).wrapping_mul(STYLE_DOMAIN);
    splitmix64(&mut state)
}

/// One step of splitmix64; also the noise stream of a styled policy.
pub(crate) fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut mixed = *state;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^ (mixed >> 31)
}

const _: () = assert!(NOISE_KNOBS.len() < MAX_SPEC_KNOBS);
const _: () = assert!(
    NOISE_KNOBS.len() + crate::teacher::STYLE_KNOBS.len() <= MAX_SPEC_KNOBS
        && NOISE_KNOBS.len() + crate::scripted::harass_push::STYLE_KNOBS.len() <= MAX_SPEC_KNOBS
);
