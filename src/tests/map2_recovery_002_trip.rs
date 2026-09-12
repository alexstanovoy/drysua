use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Place {
    Barracks,
    Lane,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Threat {
    None,
    Escapable,
    Finish,
}
#[derive(Clone, Copy, Debug)]
pub(super) struct Setup {
    pub seed: u64,
    pub side: usize,
    pub place: Place,
    pub threat: Threat,
    pub variant: u32,
}
#[derive(Clone, Copy)]
pub(super) enum Controller<'a> {
    Direct,
    Reference,
    Neural(&'a PolicyModel),
}

#[derive(Debug, Default)]
pub(super) struct Trace {
    pub arrival: Option<u32>,
    pub restored: Option<u32>,
    pub departure: Option<u32>,
    pub ticks: u32,
    pub sent: u32,
    pub finish: bool,
    pub died: bool,
    pub hp: i32,
    pub mana: i32,
    pub path: f64,
    pub reversals: u32,
    pub stationary: u32,
    pub rows: Vec<String>,
    previous: Option<Vec2>,
    previous_delta: Option<(i64, i64)>,
    baseline_deaths: u16,
}

pub(super) fn setups(training: bool, variants: &[u32]) -> Vec<Setup> {
    let base = if training { 10098000 } else { 10099000 };
    [Place::Barracks, Place::Lane]
        .into_iter()
        .enumerate()
        .flat_map(|(place_index, place)| {
            [Threat::None, Threat::Escapable, Threat::Finish]
                .into_iter()
                .enumerate()
                .flat_map(move |(threat_index, threat)| {
                    variants.iter().flat_map(move |variant| {
                        (0..2).map(move |side| Setup {
                            seed: base
                                + place_index as u64 * 64
                                + threat_index as u64 * 16
                                + u64::from(*variant) * 2
                                + side as u64,
                            side,
                            place,
                            threat,
                            variant: *variant,
                        })
                    })
                })
        })
        .collect()
}

pub(super) fn create(setup: Setup) -> Game {
    let (mut arena, mut start) = Arena::new(ArenaConfig {
        seats: 2,
        map: MapId(2),
        seed: setup.seed,
    })
    .unwrap();
    let step = arena.configure_for_test(|world| configure(world, setup));
    for (messages, fresh) in start.messages.iter_mut().zip(step.messages) {
        messages.truncate(1);
        messages.extend(fresh);
    }
    Game {
        arena,
        seats: setup_seats(start).unwrap(),
        side: setup.side,
        total: Metrics::default(),
        terminal: false,
    }
}

fn configure(world: &mut World, setup: Setup) {
    let level = if setup.variant < 2 { 1 } else { 3 };
    world.tick = 1201 + setup.variant * 300;
    for side in 0..2 {
        let hero = world.seats[side].unit.unwrap();
        world.seats[side].gold = 0;
        world.seats[side].level = level;
        world.seats[side].xp = rules::XP_THRESHOLDS[level as usize - 1];
        world.level.insert(hero, Level(level));
        world.statuses.remove(hero);
        world.set_order(hero, UnitOrder::Stand);
        assert!(world.learn(hero, 0, &mut Vec::new()));
        if level == 3 {
            assert!(world.learn(hero, 3, &mut Vec::new()));
            assert!(world.learn(hero, 4, &mut Vec::new()));
        }
    }
    world.settle();
    for side in 0..2 {
        world.fill_pools(world.seats[side].unit.unwrap());
    }
    let hero = world.seats[setup.side].unit.unwrap();
    let direction = if setup.side == 0 { 1 } else { -1 };
    let base = match setup.place {
        Place::Barracks => {
            world.map.barracks[setup.side][0].2 + Vec2::from_ints(180 * direction, 180 * direction)
        }
        Place::Lane => Vec2::from_ints(9216 - 1200 * direction, 9216 - 1200 * direction),
    };
    let position = base
        + Vec2::from_ints(
            setup.variant as i32 * 32 * direction,
            -(setup.variant as i32) * 32 * direction,
        );
    world.transform.get_mut(hero).unwrap().pos = position;
    world.health.get_mut(hero).unwrap().hp = Fixed::from_int(if level == 1 { 100 } else { 140 });
    world.mana.get_mut(hero).unwrap().mana = Fixed::ZERO;
    world.inventory.get_mut(hero).unwrap().slots[0] =
        ItemStack::bought(ItemId(8), world.seats[setup.side].slot, world.tick);
    if setup.threat != Threat::None {
        let enemy = world.seats[1 - setup.side].unit.unwrap();
        let offset = if setup.threat == Threat::Finish {
            128
        } else {
            650
        };
        world.transform.get_mut(enemy).unwrap().pos =
            position + Vec2::from_ints(offset * direction, offset * direction);
        if setup.threat == Threat::Finish {
            world.health.get_mut(enemy).unwrap().hp = Fixed::from_int(20);
        }
    }
    assert_start_clear(world, hero);
}

fn assert_start_clear(world: &World, hero: bota_server::game::Entity) {
    let position = world.transform.get(hero).unwrap().pos;
    assert!(
        world.grid.walkable(position),
        "trip fixture starts on blocked terrain"
    );
    let radius = world.hull.get(hero).unwrap().radius;
    for entity in world.entities.iter().filter(|entity| *entity != hero) {
        if let (Some(transform), Some(hull)) = (world.transform.get(entity), world.hull.get(entity))
        {
            assert!(
                !position.within(transform.pos, radius + hull.radius),
                "trip fixture starts inside another body"
            );
        }
    }
}

pub(super) fn fountain(space: &ActionSpace) -> (usize, Vec2) {
    let (index, point) = space
        .point_candidates()
        .iter()
        .enumerate()
        .find(|(index, point)| {
            point.source == crate::PointSource::BuildingLanding(UnitKind::Fountain)
                && space.move_point_mask(ControlledUnit::Hero)[*index]
        })
        .expect("existing legal M17 fountain landing");
    (index, point.position)
}

pub(super) fn reference(state: &Game, space: &ActionSpace) -> StructuredAction {
    let seat = &state.seats[state.side];
    let Some(hero) = seat.tracker.own_hero() else {
        return StructuredAction::Continue;
    };
    let (index, goal) = fountain(space);
    let active = seat.local.active_order();
    let active_fountain = active.is_some_and(|order| matches!(order.target, crate::ActivePolicyTarget::Point(point) if distance(point, goal) < 1.0) && order.kind == ActionKind::MovePoint);
    let full = hero.hp >= hero.max_hp && hero.mana >= hero.max_mana;
    if active_fountain && (hero.pos != goal || !full) {
        return StructuredAction::Continue;
    }
    if !active_fountain {
        let enemy = space
            .entity_candidates()
            .iter()
            .enumerate()
            .find(|(_, unit)| {
                unit.kind == UnitKind::Hero
                    && unit.relation == crate::EntityRelation::Enemy
                    && unit.unit().hp <= hero.attack_damage / 2
            });
        if let Some((target, enemy)) = enemy {
            if active.is_some_and(|order| {
                order.kind == ActionKind::AttackUnit
                    && order.target == crate::ActivePolicyTarget::Unit(enemy.id())
            }) {
                return StructuredAction::Continue;
            }
            let attack = StructuredAction::AttackUnit {
                unit: ControlledUnit::Hero,
                target: EntityIndex(target),
            };
            if space.allows(attack) {
                return attack;
            }
        }
    }
    if !full && distance(hero.pos, goal) < 1.0 {
        return StructuredAction::Continue;
    }
    if hero.hp * 2 < hero.max_hp || hero.mana * 3 < hero.max_mana {
        return StructuredAction::MovePoint {
            unit: ControlledUnit::Hero,
            point: PointIndex(index),
        };
    }
    if active.is_some_and(|order| matches!(order.target, crate::ActivePolicyTarget::Point(point) if distance(hero.pos, point) > 1.0)) { return StructuredAction::Continue; }
    let point = space
        .point_candidates()
        .iter()
        .enumerate()
        .filter(|(index, _)| space.move_point_mask(ControlledUnit::Hero)[*index])
        .min_by_key(|(_, point)| point.position.distance_squared(Vec2::from_ints(9216, 9216)))
        .unwrap()
        .0;
    StructuredAction::MovePoint {
        unit: ControlledUnit::Hero,
        point: PointIndex(point),
    }
}

pub(super) fn valid_demonstration(setup: Setup, trace: &Trace) -> bool {
    if setup.threat == Threat::Finish {
        trace.finish && !trace.died
    } else {
        trace.success()
    }
}

pub(super) fn act(
    state: &mut Game,
    action: StructuredAction,
    opponent: bool,
    mut observe: impl FnMut(&Game),
) {
    let (_, space) = state.prepare();
    let request = neural_policy_request_in_space(&mut state.seats[state.side], action, &space)
        .unwrap()
        .1;
    let enemy = if opponent {
        teacher_request(&mut state.seats[1 - state.side]).unwrap()
    } else {
        None
    };
    for tick in 0..3 {
        if state.terminal {
            break;
        }
        state.tick(
            if tick == 0 { request } else { None },
            if tick == 0 { enemy } else { None },
        );
        observe(state);
    }
}

impl Trace {
    pub(super) fn success(&self) -> bool {
        !self.died && self.arrival.is_some() && self.restored.is_some() && self.departure.is_some()
    }
    pub(super) fn observe(&mut self, state: &Game, start: u32, goal: Vec2) {
        self.ticks = state.arena.tick() - start;
        let tracker = &state.seats[state.side].tracker;
        self.died |= tracker.own_player().unwrap().deaths > self.baseline_deaths;
        self.finish |= tracker.own_player().unwrap().kills > 0;
        let Some(hero) = tracker.own_hero() else {
            self.previous = None;
            self.previous_delta = None;
            return;
        };
        self.hp = hero.hp;
        self.mana = hero.mana;
        if let Some(before) = self.previous {
            let delta = (
                i64::from(hero.pos.x.raw) - i64::from(before.x.raw),
                i64::from(hero.pos.y.raw) - i64::from(before.y.raw),
            );
            self.path += distance(before, hero.pos);
            self.stationary += u32::from(delta == (0, 0));
            if delta != (0, 0) {
                if let Some(last) = self.previous_delta {
                    self.reversals += u32::from(last.0 * delta.0 + last.1 * delta.1 < 0);
                }
                self.previous_delta = Some(delta);
            }
        }
        self.previous = Some(hero.pos);
        if self.died {
            return;
        }
        if hero.pos == goal && self.arrival.is_none() {
            self.arrival = Some(self.ticks);
        }
        if self.arrival.is_some()
            && hero.hp >= hero.max_hp
            && hero.mana >= hero.max_mana
            && self.restored.is_none()
        {
            self.restored = Some(self.ticks);
        }
        if self.restored.is_some()
            && distance(hero.pos, goal) >= 1200.0
            && distance(hero.pos, Vec2::from_ints(9216, 9216))
                < distance(goal, Vec2::from_ints(9216, 9216)) - 1000.0
            && self.departure.is_none()
        {
            self.departure = Some(self.ticks);
        }
    }
}

pub(super) fn run(setup: Setup, controller: Controller<'_>) -> (Trace, Metrics) {
    let mut state = create(setup);
    let start = state.arena.tick();
    let (_, space) = state.prepare();
    let (point, goal) = fountain(&space);
    let mut trace = Trace {
        previous: state.seats[setup.side]
            .tracker
            .own_hero()
            .map(|hero| hero.pos),
        ..Trace::default()
    };
    for decision in 0..600 {
        if state.terminal {
            break;
        }
        let (frame, space) = state.prepare();
        let action = match controller {
            Controller::Direct if decision == 0 => StructuredAction::MovePoint {
                unit: ControlledUnit::Hero,
                point: PointIndex(point),
            },
            Controller::Direct => StructuredAction::Continue,
            Controller::Reference => reference(&state, &space),
            Controller::Neural(model) => model.choose(&frame, &space).unwrap().action,
        };
        if trace.rows.len() < 96 && (action != StructuredAction::Continue || decision % 30 == 0) {
            trace.rows.push(format!(
                "tick={} action={action:?} active={:?} hero={:?}",
                state.arena.tick(),
                state.seats[setup.side].local.active_order(),
                state.seats[setup.side]
                    .tracker
                    .own_hero()
                    .map(|hero| (hero.pos, hero.hp, hero.mana))
            ));
        }
        let sent = state.seats[setup.side].sequence;
        act(&mut state, action, setup.threat != Threat::None, |state| {
            trace.observe(state, start, goal)
        });
        trace.sent += state.seats[setup.side].sequence - sent;
        if trace.success() || setup.threat == Threat::Finish && trace.finish && !trace.died {
            break;
        }
    }
    (trace, state.total)
}

pub(super) fn distance(source: Vec2, target: Vec2) -> f64 {
    ((f64::from(source.x.raw) - f64::from(target.x.raw)) / 65536.0)
        .hypot((f64::from(source.y.raw) - f64::from(target.y.raw)) / 65536.0)
}

#[test]
#[ignore = "Guarded native calibration before creating recovery labels; not full games."]
fn calibrate_recovery_controls() {
    let frozen = parent();
    let mut report = String::new();
    for setup in setups(true, &[0]) {
        for (name, controller) in [
            ("move_keep", Controller::Direct),
            ("state_reference", Controller::Reference),
            ("parent_nn", Controller::Neural(&frozen)),
        ] {
            let (mut trace, metrics) = run(setup, controller);
            let rows = std::mem::take(&mut trace.rows);
            writeln!(report, "setup={setup:?} controller={name} successful_trip={} trace={trace:?} metrics={metrics:?}", trace.success()).unwrap();
            for row in rows {
                writeln!(report, "{row}").unwrap();
            }
        }
    }
    for side in 0..2 {
        let physics = Physics {
            side,
            own_hp: 100,
            enemy_hp: 680,
            mana: 353,
            mango: false,
            facing: if side == 0 { 0 } else { 32768 },
            ..mango_physics()
        };
        let mut successes = 0;
        for name in ["move_keep", "state_reference", "parent_nn"] {
            let mut state = game(physics);
            let start = state.arena.tick();
            let (_, space) = state.prepare();
            let (point, goal) = fountain(&space);
            let mut trace = Trace {
                baseline_deaths: 1,
                ..Trace::default()
            };
            for decision in 0..600 {
                if state.terminal {
                    break;
                }
                let (frame, space) = state.prepare();
                let action = match name {
                    "move_keep" if decision == 0 => StructuredAction::MovePoint {
                        unit: ControlledUnit::Hero,
                        point: PointIndex(point),
                    },
                    "move_keep" => StructuredAction::Continue,
                    "state_reference" => reference(&state, &space),
                    _ => frozen.choose(&frame, &space).unwrap().action,
                };
                act(&mut state, action, true, |state| {
                    trace.observe(state, start, goal)
                });
            }
            successes += usize::from(trace.success());
            writeln!(
                report,
                "negative_mid side={side} controller={name} trace={trace:?} metrics={:?}",
                state.total
            )
            .unwrap();
        }
        writeln!(report, "negative_mid side={side} successful_tested_references={successes}; if_zero=no_successful_tested_reference_not_globally_unwinnable").unwrap();
    }
    training::write_new(&output().join("CALIBRATION.txt"), &report);
    eprintln!("{report}");
}
