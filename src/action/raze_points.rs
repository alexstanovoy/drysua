//! Point candidates that make area and blind razes expressible, and the raze
//! coverage every point candidate carries.
//!
//! A raze point only chooses a heading: the landing sits at the raze's fixed
//! reach on the line toward the point. These candidates follow the general ones
//! and are legal only as raze targets, so movement and item targeting keep
//! exactly the general candidates.

use bota_proto::{EntityId, Fixed, StatusFlags, Team, UnitKind, UnitView, Vec2, WorldView};

use super::{
    ActionError, PointCandidate, PointSource, StaticPassability, canonical_position_key,
    clamp_position, has_status, push_point,
};
use crate::StateTracker;
use crate::raze_aim::{
    SHADOWRAZE_RADIUS, SHADOWRAZES, extrapolated_position, facing_towards, raze_center,
    within_reach_window,
};

/// Raze-only candidates appended after the general point candidates.
pub(super) const RAZE_POINT_CANDIDATES: usize = 16;
/// Distance of the facing and blind-direction candidates: the mid raze reach.
pub(crate) const RAZE_RING_RADIUS: i32 = 450;
/// Headings scanned for each reach's best landing, 512 brads (34 units at 700) apart.
const CLUSTER_HEADINGS: u16 = 128;
const CLUSTER_STEP: u16 = (65_536 / CLUSTER_HEADINGS as u32) as u16;
/// Eight blind headings between the eight tactical ones complete a 16-direction ring.
const RING_HEADINGS: u16 = 8;
/// Fogged enemy heroes that get a last-seen and an extrapolated guess.
const FOG_HEROES: usize = 2;
/// Sightings older than this are not guessed at.
pub(crate) const FOG_GUESS_MAX_AGE_TICKS: u32 = 150;

const _: () = assert!(
    1 + SHADOWRAZES.len() + 2 * FOG_HEROES + RING_HEADINGS as usize == RAZE_POINT_CANDIDATES
);
const _: () = assert!(CLUSTER_STEP as u32 * CLUSTER_HEADINGS as u32 == 65_536);

/// Visible hostile units one raze landing would strike.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct RazeCoverage {
    /// Live enemy or neutral units that are not heroes, structures included.
    pub units: u8,
    /// Live enemy heroes.
    pub heroes: u8,
}

impl RazeCoverage {
    /// Coverage of a point that gives the raze no heading or lies beyond the caster's view.
    pub const NONE: Self = Self {
        units: 0,
        heroes: 0,
    };

    fn total(self) -> u16 {
        u16::from(self.units) + u16::from(self.heroes)
    }
}

/// A visible hostile unit a raze could strike.
#[derive(Clone, Copy, Debug)]
pub(super) struct RazeTarget {
    position: Vec2,
    hero: bool,
}

/// The own hero while it could ever cast a raze.
pub(super) fn raze_caster(tracker: &StateTracker) -> Option<&UnitView> {
    tracker.own_hero().filter(|hero| hero.hp > 0)
}

/// Visible hostile units any raze landing could strike from `origin`.
pub(super) fn raze_targets(
    tracker: &StateTracker,
    current: &WorldView,
    origin: Vec2,
) -> Vec<RazeTarget> {
    // One unit of slack, as in the reach window, absorbs landing rounding.
    let outermost = Fixed::from_int(SHADOWRAZES[SHADOWRAZES.len() - 1].1 + SHADOWRAZE_RADIUS + 1);
    current
        .units
        .iter()
        .filter(|unit| {
            unit.team != tracker.team()
                && unit.hp > 0
                && !has_status(unit, StatusFlags::INVULNERABLE)
                && origin.within(unit.pos, outermost)
        })
        .map(|unit| RazeTarget {
            position: unit.pos,
            hero: unit.kind == UnitKind::Hero,
        })
        .collect()
}

/// Appends the facing, best-landing, fog-guess and blind-ring candidates in that order.
pub(super) fn add_raze_points(
    tracker: &StateTracker,
    current: &WorldView,
    passability: &StaticPassability,
    hero: &UnitView,
    targets: &[RazeTarget],
    points: &mut Vec<PointCandidate>,
) -> Result<(), ActionError> {
    let cap = points.len() + RAZE_POINT_CANDIDATES;
    let mut push = |position: Vec2, source: PointSource| -> Result<(), ActionError> {
        let position = clamp_position(tracker, position)?;
        let candidate = PointCandidate {
            position,
            source,
            walkable: passability.walkable(position),
            standing_tree: false,
            allied_building: false,
            raze_coverage: [RazeCoverage::NONE; SHADOWRAZES.len()],
        };
        push_point(points, candidate, cap);
        Ok(())
    };
    push(
        raze_center(hero.pos, hero.facing.brads, RAZE_RING_RADIUS),
        PointSource::RazeFacing,
    )?;
    let base = canonical_heading(tracker.team());
    for (_, reach) in SHADOWRAZES {
        if let Some(heading) = best_heading(hero.pos, reach, base, targets) {
            push(
                raze_center(hero.pos, heading, reach),
                PointSource::RazeCluster { reach },
            )?;
        }
    }
    for (age, unit) in fog_heroes(tracker, current) {
        push(unit.pos, PointSource::LastSeenHero { age })?;
        push(
            extrapolated_position(tracker, unit, age),
            PointSource::ExtrapolatedHero { age },
        )?;
    }
    for index in 0..RING_HEADINGS {
        let heading = base.wrapping_add(4_096).wrapping_add(index * 8_192);
        push(
            raze_center(hero.pos, heading, RAZE_RING_RADIUS),
            PointSource::RazeRing,
        )?;
    }
    Ok(())
}

/// Sets every candidate's coverage for each raze reach along the heading toward it.
pub(super) fn fill_raze_coverage(
    origin: Vec2,
    targets: &[RazeTarget],
    points: &mut [PointCandidate],
) {
    for point in points.iter_mut().filter(|point| point.position != origin) {
        let heading = facing_towards(origin, point.position);
        for (coverage, (_, reach)) in point.raze_coverage.iter_mut().zip(SHADOWRAZES) {
            *coverage = coverage_at(raze_center(origin, heading, reach), targets);
        }
    }
}

fn coverage_at(landing: Vec2, targets: &[RazeTarget]) -> RazeCoverage {
    let radius = Fixed::from_int(SHADOWRAZE_RADIUS);
    let mut coverage = RazeCoverage::NONE;
    for target in targets
        .iter()
        .filter(|target| landing.within(target.position, radius))
    {
        let count = if target.hero {
            &mut coverage.heroes
        } else {
            &mut coverage.units
        };
        *count = count.saturating_add(1);
    }
    coverage
}

/// The team-canonical east: Dire headings are the Radiant ones turned half round.
const fn canonical_heading(team: Team) -> u16 {
    if matches!(team, Team::Dire) {
        32_768
    } else {
        0
    }
}

/// Middle of the widest run of scanned headings whose landing strikes the most
/// hostile units (then heroes); nothing when no heading strikes any.
fn best_heading(origin: Vec2, reach: i32, base: u16, targets: &[RazeTarget]) -> Option<u16> {
    let reachable = targets
        .iter()
        .copied()
        .filter(|target| within_reach_window(origin, target.position, reach))
        .collect::<Vec<_>>();
    if reachable.is_empty() {
        return None;
    }
    let heading = |index: u16| base.wrapping_add(index.wrapping_mul(CLUSTER_STEP));
    let scores: [(u16, u8); CLUSTER_HEADINGS as usize] = std::array::from_fn(|index| {
        let coverage = coverage_at(
            raze_center(origin, heading(index as u16), reach),
            &reachable,
        );
        (coverage.total(), coverage.heroes)
    });
    let best = scores.iter().copied().max()?;
    if best.0 == 0 {
        return None;
    }
    let count = CLUSTER_HEADINGS as usize;
    if scores.iter().all(|score| *score == best) {
        return Some(heading(0));
    }
    // Runs start where the previous heading is worse; the first widest run wins.
    let mut widest = (0usize, 0usize);
    for start in 0..count {
        if scores[start] != best || scores[(start + count - 1) % count] == best {
            continue;
        }
        let length = (0..count)
            .take_while(|offset| scores[(start + offset) % count] == best)
            .count();
        if length > widest.1 {
            widest = (start, length);
        }
    }
    let (start, length) = widest;
    assert!(length > 0 && length < count);
    let middle =
        u32::from(heading(start as u16)) + (length as u32 - 1) * u32::from(CLUSTER_STEP) / 2;
    Some(middle as u16)
}

/// Enemy heroes out of sight, not known dead, seen at most the guess age ago,
/// freshest first; each with its sighting age in ticks.
fn fog_heroes<'a>(tracker: &'a StateTracker, current: &WorldView) -> Vec<(u32, &'a UnitView)> {
    let visible = |id: EntityId| {
        current
            .units
            .binary_search_by_key(&id, |unit| unit.id)
            .is_ok()
    };
    let mut heroes = tracker
        .entities()
        .iter()
        .filter(|track| {
            track.unit.kind == UnitKind::Hero
                && track.unit.team != tracker.team()
                && track.unit.team != Team::Neutral
                && track.unit.hp > 0
                && track
                    .last_death
                    .is_none_or(|death| death.tick < track.last_seen_tick)
                && !visible(track.id)
        })
        .filter_map(|track| {
            let age = current.tick.checked_sub(track.last_seen_tick)?;
            (age <= FOG_GUESS_MAX_AGE_TICKS).then_some((age, &track.unit))
        })
        .collect::<Vec<_>>();
    heroes.sort_by_key(|(age, unit)| (*age, canonical_position_key(tracker, unit.pos), unit.id));
    heroes.truncate(FOG_HEROES);
    heroes
}
