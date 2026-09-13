use bota_proto::{EntityId, EventKind, Fixed, UnitKind, Vec2};

use super::observation::{SnapshotFacts, UnitFact};
use super::{
    MAP2_REWARD_FOUNTAIN_BASE_COST, MAP2_REWARD_FOUNTAIN_COST_PER_SECOND,
    MAP2_REWARD_FOUNTAIN_GRACE_TICKS, MAP2_REWARD_FOUNTAIN_WAIT_BOUND,
    MAP2_REWARD_PREGAME_CENTER_SCALE, MAX_TICK, Map2Reward,
};

const MAP2_CENTER: f64 = 9216.0;
const FOUNTAIN_RADIUS: i128 = 1200_i128 << Fixed::FRAC_BITS;

impl Map2Reward {
    pub(super) fn observe_pregame_movement(&mut self, pending: &SnapshotFacts) {
        if pending.tick >= self.pregame_ticks {
            return;
        }
        let Some(hero) = own_body(pending) else {
            return;
        };
        let potential = center_potential(hero.pos);
        if self.current.is_some()
            && let Some(previous) = self.pregame_center_potential
        {
            self.interval.pregame_movement += potential - previous;
        }
        self.pregame_center_potential = Some(potential);
        assert!(potential >= 0.0);
        assert!(potential <= MAP2_REWARD_PREGAME_CENTER_SCALE);
    }

    pub(super) fn observe_fountain_wait(&mut self, pending: &SnapshotFacts, events: &[EventKind]) {
        let own_purchase = events.iter().any(|event| {
            matches!(event,
            EventKind::ItemBought { slot, .. } if *slot == self.roles[0].slot)
        });
        if own_purchase {
            self.interval.fountain_wait_refund += self.fountain_wait_refundable_cost;
            self.interval.observations.fountain_wait_refunds +=
                u64::from(self.fountain_wait_refundable_cost > 0.0);
            self.reset_fountain_wait();
            return;
        }
        let current = self.full_fountain_body(pending);
        let previous = self
            .current
            .as_ref()
            .and_then(|view| self.full_fountain_body(view));
        if current.is_none() || current != previous {
            self.reset_fountain_wait();
            return;
        }
        self.fountain_wait_ticks += 1;
        let cost = wait_cost(self.fountain_wait_ticks);
        let increment = cost - self.fountain_wait_refundable_cost;
        assert!(increment >= 0.0);
        assert!(cost <= MAP2_REWARD_FOUNTAIN_WAIT_BOUND);
        self.interval.fountain_wait -= increment;
        self.interval.observations.fountain_wait_ticks += 1;
        self.interval.observations.fountain_wait_charged_ticks += u64::from(increment > 0.0);
        self.fountain_wait_refundable_cost = cost;
    }

    fn full_fountain_body(&self, view: &SnapshotFacts) -> Option<(EntityId, Vec2)> {
        let hero = own_body(view)?;
        let mana = view.mana?;
        assert_eq!(hero.id, mana.id);
        if hero.hp != hero.maximum || mana.mana != mana.maximum {
            return None;
        }
        view.units
            .iter()
            .any(|unit| {
                unit.kind == UnitKind::Fountain
                    && unit.team == self.roles[0].team
                    && in_fountain(hero.pos, unit.pos)
            })
            .then_some((hero.id, hero.pos))
    }

    fn reset_fountain_wait(&mut self) {
        self.fountain_wait_ticks = 0;
        self.fountain_wait_refundable_cost = 0.0;
    }
}

fn own_body(view: &SnapshotFacts) -> Option<&UnitFact> {
    let id = view.heroes[0]?;
    let index = view.units.binary_search_by_key(&id, |unit| unit.id).ok()?;
    let hero = &view.units[index];
    (hero.hp > 0).then_some(hero)
}

fn center_potential(position: Vec2) -> f64 {
    let scale = f64::from(Fixed::ONE.raw);
    let x = f64::from(position.x.raw) / scale - MAP2_CENTER;
    let y = f64::from(position.y.raw) / scale - MAP2_CENTER;
    let fraction = ((x * x + y * y) / (2.0 * MAP2_CENTER * MAP2_CENTER)).sqrt();
    MAP2_REWARD_PREGAME_CENTER_SCALE * (1.0 - fraction.clamp(0.0, 1.0))
}

fn in_fountain(hero: Vec2, fountain: Vec2) -> bool {
    let x = i128::from(hero.x.raw) - i128::from(fountain.x.raw);
    let y = i128::from(hero.y.raw) - i128::from(fountain.y.raw);
    x * x + y * y <= FOUNTAIN_RADIUS * FOUNTAIN_RADIUS
}

fn wait_cost(ticks: u32) -> f64 {
    assert!(ticks <= MAX_TICK);
    if ticks < MAP2_REWARD_FOUNTAIN_GRACE_TICKS {
        return 0.0;
    }
    MAP2_REWARD_FOUNTAIN_BASE_COST
        + MAP2_REWARD_FOUNTAIN_COST_PER_SECOND * f64::from(ticks - MAP2_REWARD_FOUNTAIN_GRACE_TICKS)
            / f64::from(MAP2_REWARD_FOUNTAIN_GRACE_TICKS)
}
