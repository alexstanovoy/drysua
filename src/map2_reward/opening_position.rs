use bota_proto::{Fixed, Vec2};

use super::observation::SnapshotFacts;
use super::{MAP2_REWARD_OPENING_POSITION_BOUND, Map2Reward};

const CENTER: i128 = 9216_i128 << Fixed::FRAC_BITS;
const FREE_RADIUS: i128 = 1500_i128 << Fixed::FRAC_BITS;
const FULL_RADIUS: i128 = 3000_i128 << Fixed::FRAC_BITS;
const _: () = assert!(FREE_RADIUS < FULL_RADIUS);

impl Map2Reward {
    pub(super) fn observe_opening_position(&mut self, pending: &SnapshotFacts) {
        if !self.opening_position_pending {
            return;
        }
        let first_wave =
            pending.tick >= self.pregame_ticks && pending.tick < self.opening_position_deadline;
        let approaching = first_wave
            && pending.units.iter().any(|unit| {
                unit.team == self.roles[0].team
                    && unit.hp > 0
                    && !unit.dead
                    && super::events::lane_creep(unit.kind)
                    && distance_squared(unit.pos) <= FREE_RADIUS * FREE_RADIUS
            });
        if pending.tick < self.opening_position_deadline && !approaching {
            return;
        }
        self.opening_position_pending = false;
        // An already-observed trigger on a late initial baseline is waived, not deferred.
        if self.current.is_none() {
            return;
        }
        let body = pending.heroes[0]
            .and_then(|id| {
                pending
                    .units
                    .binary_search_by_key(&id, |unit| unit.id)
                    .ok()
                    .map(|index| &pending.units[index])
            })
            .filter(|unit| unit.hp > 0 && !unit.dead);
        self.interval.opening_position -= body.map_or(MAP2_REWARD_OPENING_POSITION_BOUND, |unit| {
            position_cost(unit.pos)
        });
        self.interval.observations.opening_position_checks += 1;
        assert!(self.interval.opening_position >= -MAP2_REWARD_OPENING_POSITION_BOUND);
        assert_eq!(self.interval.observations.opening_position_checks, 1);
    }
}

fn distance_squared(position: Vec2) -> i128 {
    let x = i128::from(position.x.raw) - CENTER;
    let y = i128::from(position.y.raw) - CENTER;
    x * x + y * y
}

fn position_cost(position: Vec2) -> f64 {
    let squared = distance_squared(position);
    if squared <= FREE_RADIUS * FREE_RADIUS {
        return 0.0;
    }
    if squared >= FULL_RADIUS * FULL_RADIUS {
        return MAP2_REWARD_OPENING_POSITION_BOUND;
    }
    let distance = (squared as f64).sqrt();
    let fraction = (distance - FREE_RADIUS as f64) / FREE_RADIUS as f64;
    assert!((0.0..=1.0).contains(&fraction));
    MAP2_REWARD_OPENING_POSITION_BOUND * fraction
}
