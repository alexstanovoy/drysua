use super::*;

#[derive(Debug)]
pub(super) struct Outcome {
    pub exact: trip::Trace,
    pub entered: Option<u32>,
    pub full: Option<u32>,
    pub departed: Option<u32>,
    fountain: Vec2,
    full_position: Option<Vec2>,
}

impl Outcome {
    pub(super) fn new(state: &Game) -> Self {
        let tracker = &state.seats[state.side].tracker;
        let fountain = tracker
            .current()
            .unwrap()
            .units
            .iter()
            .find(|unit| unit.kind == UnitKind::Fountain && unit.team == tracker.team())
            .unwrap()
            .pos;
        Self {
            exact: trip::Trace::default(),
            entered: None,
            full: None,
            departed: None,
            fountain,
            full_position: None,
        }
    }
    pub(super) fn success(&self) -> bool {
        !self.exact.died && self.entered.is_some() && self.full.is_some() && self.departed.is_some()
    }
    pub(super) fn observe(&mut self, state: &Game, start: u32, landing: Vec2) {
        self.exact.observe(state, start, landing);
        if self.exact.died {
            return;
        }
        if let Some(hero) = state.seats[state.side].tracker.own_hero() {
            self.observe_pools(state.arena.tick() - start, hero);
        }
    }
    pub(super) fn observe_pools(&mut self, tick: u32, hero: &bota_proto::UnitView) {
        assert!(hero.max_hp > 0);
        assert!(hero.max_mana > 0);
        let inside = hero
            .pos
            .within(self.fountain, Fixed::from_int(rules::FOUNTAIN_HEAL_RADIUS));
        if inside && self.entered.is_none() {
            self.entered = Some(tick);
        }
        if inside && hero.hp >= hero.max_hp && hero.mana >= hero.max_mana && self.full.is_none() {
            self.full = Some(tick);
            self.full_position = Some(hero.pos);
        }
        if let Some(position) = self.full_position
            && !inside
            && trip::distance(position, Vec2::from_ints(9216, 9216))
                - trip::distance(hero.pos, Vec2::from_ints(9216, 9216))
                >= 1000.0
            && self.departed.is_none()
        {
            self.departed = Some(tick);
        }
    }
}

pub(super) fn valid(setup: trip::Setup, outcome: &Outcome) -> bool {
    if setup.threat == trip::Threat::Finish {
        outcome.exact.finish && !outcome.exact.died
    } else {
        outcome.success()
    }
}

pub(super) fn run(setup: trip::Setup, model: &PolicyModel) -> (Outcome, Metrics) {
    let mut state = trip::create(setup);
    let start = state.arena.tick();
    let (_, space) = state.prepare();
    let (_, landing) = trip::fountain(&space);
    let mut outcome = Outcome::new(&state);
    outcome.observe(&state, start, landing);
    for decision in 0..600 {
        if state.terminal {
            break;
        }
        let (frame, space) = state.prepare();
        let action = model.choose(&frame, &space).unwrap().action;
        if outcome.exact.rows.len() < 64
            && (action != StructuredAction::Continue || decision % 30 == 0)
        {
            outcome.exact.rows.push(format!(
                "tick={} action={action:?} active={:?} hero={:?}",
                state.arena.tick(),
                state.seats[setup.side].local.active_order(),
                state.seats[setup.side].tracker.own_hero().map(|hero| (
                    hero.pos,
                    hero.hp,
                    hero.mana,
                    hero.max_hp,
                    hero.max_mana
                ))
            ));
        }
        let sequence = state.seats[setup.side].sequence;
        trip::act(
            &mut state,
            action,
            setup.threat != trip::Threat::None,
            |state| outcome.observe(state, start, landing),
        );
        outcome.exact.sent += state.seats[setup.side].sequence - sequence;
        if valid(setup, &outcome) {
            break;
        }
    }
    (outcome, state.total)
}

pub(super) fn controlled_endpoint(early: bool) -> (Outcome, Vec2) {
    let setup = trip::Setup {
        seed: 10100090,
        side: 0,
        place: trip::Place::Barracks,
        threat: trip::Threat::None,
        variant: 0,
    };
    let mut state = trip::create(setup);
    let start = state.arena.tick();
    let (_, space) = state.prepare();
    let (index, landing) = trip::fountain(&space);
    let mut outcome = Outcome::new(&state);
    let mut departed_from = None;
    for decision in 0..600 {
        let (_, space) = state.prepare();
        let ready = if early {
            outcome.full.is_some()
        } else {
            outcome.exact.restored.is_some()
        };
        let action = if decision == 0 {
            StructuredAction::MovePoint {
                unit: ControlledUnit::Hero,
                point: PointIndex(index),
            }
        } else if ready && departed_from.is_none() {
            departed_from = Some(state.seats[0].tracker.own_hero().unwrap().pos);
            let point = space
                .point_candidates()
                .iter()
                .enumerate()
                .filter(|(index, _)| space.move_point_mask(ControlledUnit::Hero)[*index])
                .min_by_key(|(_, point)| {
                    point.position.distance_squared(Vec2::from_ints(9216, 9216))
                })
                .unwrap()
                .0;
            StructuredAction::MovePoint {
                unit: ControlledUnit::Hero,
                point: PointIndex(point),
            }
        } else {
            StructuredAction::Continue
        };
        assert!(space.allows(action));
        trip::act(&mut state, action, false, |state| {
            outcome.observe(state, start, landing)
        });
        if outcome.success() {
            break;
        }
    }
    assert!(outcome.success());
    (outcome, departed_from.unwrap())
}
