//! Seat-visible progress watchdog: a pursued spot that stops getting closer is abandoned.
//!
//! Point candidates only approximate what the simulator's planner can reach: a walk to a
//! spot shut in by a building or trees ends at the nearest open spot short of it, a walk
//! out of a forest pocket toward a spot beyond it never starts, and an item aimed at a
//! tree that is not in reach does nothing. None of this is reported, so a policy that
//! keeps pursuing such a spot freezes. The watchdog judges progress from the hero's own
//! snapshot only and keeps a short bounded memory of what failed.

use bota_proto::{Fixed, StatusFlags, UnitView, Vec2};

use crate::StateTracker;
use crate::raze_aim::isqrt;
use crate::scripted::tactics::DECISION_TICKS;

/// Pursued ticks without progress after which a spot counts as unreachable; longer than
/// a full turn and a creep brushing past.
const STALL_TICKS: u32 = 30;
/// Distance a pursuit must gain on its best approach to count as progress.
const PROGRESS_UNITS: i32 = 32;
/// How long an unreachable spot is avoided before it may be tried again.
const AVOID_TICKS: u32 = 1_800;
/// How long a dead end is avoided: long enough to walk out and pick another way.
const DEAD_END_TICKS: u32 = 450;
/// Spots this close to an unreachable one are avoided with it.
const TARGET_RADIUS: i32 = 64;
/// Covers every tactical point around one boxed-in position.
const AVOID_LIMIT: usize = 32;
/// Stalls toward different spots from one position that prove the position a dead end.
const DEAD_END_STALLS: u8 = 3;
/// Stalls this close together count as made from one position.
const SAME_POSITION_UNITS: i32 = 64;
/// Around a dead end nothing is pursued, so the hero does not walk straight back in.
const DEAD_END_RADIUS: i32 = 300;
/// Spacing of the breadcrumbs the hero leaves; the older one is a spot it walked from.
const TRAIL_UNITS: i32 = 400;
/// A jump this long between two decisions is a respawn or teleport, not a walk.
const TELEPORT_UNITS: i32 = 1_500;

/// A spot the hero is currently sent toward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Pursuit {
    pub target: Vec2,
    /// Standing within this many units of the target completes the pursuit.
    pub arrival: Option<i32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Watch {
    target: Vec2,
    closest: u64,
    /// Ticks the target was pursued since the last progress. Gaps are not counted, so a
    /// pursuit re-issued after a stop keeps its history.
    stalled: u32,
    seen: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Avoided {
    center: Vec2,
    radius: i32,
    until: u32,
}

/// Bounded memory of the latest pursuit, the hero's trail and what recently failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GoalProgress {
    watch: Option<Watch>,
    avoided: [Option<Avoided>; AVOID_LIMIT],
    /// Where the latest stalls happened and how many in a row happened there.
    stalls: Option<(Vec2, u8)>,
    /// The newest breadcrumb and the one before it.
    trail: Option<(Vec2, Option<Vec2>)>,
    /// A dead end the hero is still in, the breadcrumb that leads out and a deadline.
    escape: Option<(Vec2, Vec2, u32)>,
}

impl GoalProgress {
    pub(crate) const fn new() -> Self {
        Self {
            watch: None,
            avoided: [None; AVOID_LIMIT],
            stalls: None,
            trail: None,
            escape: None,
        }
    }

    /// Advances the watch on `pursuit` and the trail by one observed snapshot.
    pub(crate) fn observe(&mut self, tracker: &StateTracker, pursuit: Option<Pursuit>) {
        let (Some(hero), Some(view)) = (tracker.own_hero(), tracker.current()) else {
            self.watch = None;
            return;
        };
        let now = view.tick;
        for spot in &mut self.avoided {
            if spot.is_some_and(|spot| spot.until <= now) {
                *spot = None;
            }
        }
        self.follow_trail(hero.pos, now);
        let Some(pursuit) = pursuit else {
            return;
        };
        if pursuit
            .arrival
            .is_some_and(|arrival| hero.pos.within(pursuit.target, Fixed::from_int(arrival)))
        {
            self.watch = None;
            return;
        }
        if self.stalled(tracker, hero, pursuit.target, now) {
            self.watch = None;
            self.note_stall(hero.pos, pursuit.target, now);
        }
    }

    /// Whether `position` is neither near a spot recently found unreachable nor in a dead end.
    pub(crate) fn allows(&self, position: Vec2) -> bool {
        self.avoided
            .iter()
            .flatten()
            .all(|spot| !spot.center.within(position, Fixed::from_int(spot.radius)))
    }

    /// A spot the hero walked from, while it is still boxed in a dead end.
    pub(crate) const fn escape(&self) -> Option<Vec2> {
        match self.escape {
            Some((_, breadcrumb, _)) => Some(breadcrumb),
            None => None,
        }
    }

    fn stalled(&mut self, tracker: &StateTracker, hero: &UnitView, target: Vec2, now: u32) -> bool {
        let mut watch = self
            .watch
            .filter(|watch| watch.target == target)
            .unwrap_or(Watch {
                target,
                closest: u64::MAX,
                stalled: 0,
                seen: now,
            });
        let distance = isqrt(hero.pos.distance_squared(target).max(0) as u64);
        if now.saturating_sub(watch.seen) > DECISION_TICKS {
            // The hero may have walked elsewhere meanwhile; judge the resumed approach afresh.
            watch.closest = watch.closest.max(distance);
        }
        let gain = Fixed::from_int(PROGRESS_UNITS).raw as u64;
        // Fighting on the way or being held in place is not a failed route.
        let held = hero.statuses.bits & (StatusFlags::STUNNED | StatusFlags::CHANNELLING) != 0;
        let fought = tracker
            .entity(hero.id)
            .and_then(|track| track.last_damage_dealt)
            .is_some_and(|damage| damage.tick > watch.seen);
        if distance.saturating_add(gain) <= watch.closest || held || fought {
            watch.closest = watch.closest.min(distance);
            watch.stalled = 0;
        } else {
            watch.stalled += now.saturating_sub(watch.seen).min(DECISION_TICKS);
        }
        watch.seen = now;
        self.watch = Some(watch);
        watch.stalled >= STALL_TICKS
    }

    fn note_stall(&mut self, position: Vec2, target: Vec2, now: u32) {
        self.avoid(target, TARGET_RADIUS, now.saturating_add(AVOID_TICKS));
        let count = match self.stalls {
            Some((at, count)) if at.within(position, Fixed::from_int(SAME_POSITION_UNITS)) => {
                count.saturating_add(1)
            }
            _ => 1,
        };
        self.stalls = Some((position, count));
        if count >= DEAD_END_STALLS {
            let until = now.saturating_add(DEAD_END_TICKS);
            self.avoid(position, DEAD_END_RADIUS, until);
            self.escape = self
                .trail
                .and_then(|(_, previous)| previous)
                .map(|breadcrumb| (position, breadcrumb, until));
        }
    }

    fn follow_trail(&mut self, position: Vec2, now: u32) {
        let step = Fixed::from_int(TRAIL_UNITS);
        self.trail = match self.trail {
            Some((newest, _)) if !newest.within(position, Fixed::from_int(TELEPORT_UNITS)) => {
                self.escape = None;
                Some((position, None))
            }
            Some((newest, previous)) if newest.within(position, step) => Some((newest, previous)),
            Some((newest, _)) => Some((position, Some(newest))),
            None => Some((position, None)),
        };
        if self.escape.is_some_and(|(dead_end, _, until)| {
            until <= now || !dead_end.within(position, Fixed::from_int(DEAD_END_RADIUS))
        }) {
            self.escape = None;
        }
    }

    fn avoid(&mut self, center: Vec2, radius: i32, until: u32) {
        // A full memory forgets the entry closest to expiry; it is retried soonest anyway.
        let slot = self
            .avoided
            .iter()
            .position(Option::is_none)
            .unwrap_or_else(|| {
                (0..AVOID_LIMIT)
                    .min_by_key(|index| self.avoided[*index].map_or(0, |spot| spot.until))
                    .expect("avoid memory is not empty")
            });
        self.avoided[slot] = Some(Avoided {
            center,
            radius,
            until,
        });
    }
}
