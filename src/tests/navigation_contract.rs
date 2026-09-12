use std::sync::LazyLock;

use bota_proto::{
    Fixed, ItemId, ItemSlot, MatchInfo, Order, Pick, RejectReason, SlotId, Target, Team, TickMode,
    UnitKind, Vec2,
};
use bota_server::game::{
    Command, ItemStack, MatchConfig, UnitOrder, World, find_path, grid_los, isqrt64, rules, wire_id,
};

use crate::{
    ActionError, ActionKind, ActionSpace, ActionTarget, ControlledUnit, FeatureEncoder,
    FeatureFrame, ItemReadiness, LocalPolicyState, PointCandidate, PointIndex, PointSource,
    StateTracker, StructuredAction, point_feature,
};

const SEED: u64 = 10_091_612;
const START_TICK: u32 = 1_201;
const SHORT_LIMIT: u32 = 600;
const FOUNTAIN_LIMIT: u32 = 1_500;
const _: () = assert!(SHORT_LIMIT < FOUNTAIN_LIMIT);
const _: () = assert!(START_TICK > rules::PREGAME_TICKS);

#[derive(Clone, Copy, Debug)]
enum Place {
    Barracks,
    Lane,
}

struct Evidence {
    denied: Vec<(ControlledUnit, usize, PointSource)>,
    landings: usize,
    trees: usize,
    closed: usize,
    non_teleport: usize,
    arrival: Option<u32>,
    fountain_regeneration: bool,
}

#[test]
fn radiant_barracks_native_walkable_landing_moves_must_be_allowed() {
    assert_landing_moves_allowed(0);
}

#[test]
fn dire_barracks_native_walkable_landing_moves_must_be_allowed() {
    assert_landing_moves_allowed(1);
}

#[test]
fn radiant_lane_native_walkable_landing_moves_must_be_allowed() {
    assert_landing_moves_allowed(2);
}

#[test]
fn dire_lane_native_walkable_landing_moves_must_be_allowed() {
    assert_landing_moves_allowed(3);
}

#[test]
fn native_one_move_then_continue_reaches_existing_fountain_pointer_and_regenerates() {
    for (index, evidence) in evidence().iter().enumerate() {
        assert!(evidence.landings > 0);
        assert!(
            evidence.arrival.is_some(),
            "fixture {index}: no native arrival"
        );
        assert!(
            evidence.fountain_regeneration,
            "fixture {index}: no fountain regeneration"
        );
    }
}

#[test]
fn closed_ground_and_standing_trees_stay_masked_and_tp_requires_building_provenance() {
    let evidence = evidence();
    assert!(evidence.iter().all(|case| case.trees > 0));
    assert!(evidence.iter().map(|case| case.closed).sum::<usize>() > 0);
    assert!(evidence.iter().all(|case| case.non_teleport > 0));
}

fn assert_landing_moves_allowed(index: usize) {
    assert!(index < 4);
    let case = &evidence()[index];
    assert!(case.landings > 0);
    assert!(
        case.denied.is_empty(),
        "Native walkable, path-connected, accepted Move pointers must be allowed; fixture {index}, masked={:?}",
        case.denied
    );
}

fn evidence() -> &'static [Evidence; 4] {
    // Share the bounded native runs across red assertions and passing controls.
    static EVIDENCE: LazyLock<[Evidence; 4]> = LazyLock::new(|| {
        [
            (0, Place::Barracks),
            (1, Place::Barracks),
            (0, Place::Lane),
            (1, Place::Lane),
        ]
        .map(|(side, place)| inspect_fixture(side, place))
    });
    &EVIDENCE
}

fn fixture(side: usize, place: Place) -> (World, MatchInfo) {
    assert!(side < 2);
    let config = MatchConfig {
        match_id: SEED,
        master_key: [0; 32],
        picks: vec![
            Pick {
                slot: SlotId(0),
                team: Team::Radiant,
                hero: crate::SHADOW_FIEND,
            },
            Pick {
                slot: SlotId(1),
                team: Team::Dire,
                hero: crate::SHADOW_FIEND,
            },
        ],
        map: crate::MAP2_ID,
        tick_rate: crate::MAP2_TICK_RATE as u16,
        mode: TickMode::Lockstep,
        ack_timeout_ticks: 150,
    };
    let mut world = World::for_match(&config, config.rng());
    world.advance(&[]);
    world.tick = START_TICK;
    let hero = world.seats[side].unit.expect("stock hero");
    let direction = if side == 0 { 1 } else { -1 };
    let position = match place {
        Place::Barracks => {
            world.map.barracks[side][0].2 + Vec2::from_ints(180 * direction, 180 * direction)
        }
        Place::Lane => Vec2::from_ints(9_216 - 1_200 * direction, 9_216 - 1_200 * direction),
    };
    world.transform.get_mut(hero).expect("hero transform").pos = position;
    for seat in 0..2 {
        for body in [world.seats[seat].unit, world.seats[seat].courier] {
            let body = body.expect("native owned body");
            world.statuses.remove(body);
            world.set_order(body, UnitOrder::Stand);
        }
    }
    world.seats[side].gold = 0;
    world.inventory.get_mut(hero).expect("hero bag").slots[0] =
        ItemStack::bought(ItemId(8), world.seats[side].slot, START_TICK);
    world.settle();
    world.health.get_mut(hero).expect("health").hp = Fixed::from_int(100);
    world.mana.get_mut(hero).expect("mana").mana = Fixed::ZERO;
    assert_eq!(world.map.id, crate::MAP2_ID);
    assert!(
        world.grid.walkable(position),
        "{side} {place:?}: closed start {position:?}"
    );
    assert_body_clearance(&world, side, ControlledUnit::Hero);
    assert_body_clearance(&world, side, ControlledUnit::Courier);
    (world, config.info())
}

fn assert_body_clearance(world: &World, side: usize, unit: ControlledUnit) {
    let body = world
        .driven_by(world.seats[side].slot, selector(world, side, unit))
        .expect("body");
    let position = world.transform.get(body).expect("position").pos;
    assert!(!world.held(body));
    if unit == ControlledUnit::Courier {
        assert!(world.stats.get(body).expect("courier stats").flies);
        assert!(world.hull.get(body).is_none());
        assert_eq!(position, world.map.fountains[side]);
        return;
    }
    let radius = world.hull.get(body).expect("hull").radius;
    assert!(world.grid.walkable(position));
    for other in world.entities.iter().filter(|other| *other != body) {
        if let (Some(at), Some(hull)) = (world.transform.get(other), world.hull.get(other)) {
            assert!(
                !position.within(at.pos, radius + hull.radius),
                "{unit:?} starts overlapping {:?} at {:?}",
                world.kind.get(other),
                at.pos
            );
        }
    }
}

fn inspect_fixture(side: usize, place: Place) -> Evidence {
    let (mut world, info) = fixture(side, place);
    let view = world.view(world.seats[side].team);
    let mut tracker = StateTracker::new(world.seats[side].slot, &info).expect("native tracker");
    tracker
        .observe_snapshot(&view)
        .expect("current native snapshot");
    tracker
        .observe_events(world.tick, &[])
        .expect("complete current baseline");
    let space = ActionSpace::from_tracker(&tracker).expect("current navigation action space");
    let frame = encode(&tracker, &space);
    assert_tp_target_kinds(&world, side, &space);
    assert_eq!(space.decode(StructuredAction::Continue), Ok(None));
    assert_eq!(space.point_candidates().len(), crate::MAX_POINT_CANDIDATES);
    let mut result = Evidence {
        denied: Vec::new(),
        landings: 0,
        trees: 0,
        closed: 0,
        non_teleport: 0,
        arrival: None,
        fountain_regeneration: false,
    };
    for (index, candidate) in space.point_candidates().iter().copied().enumerate() {
        inspect_negative(&mut world, side, &space, index, candidate, &mut result);
        if !matches!(candidate.source, PointSource::BuildingLanding(_)) {
            continue;
        }
        assert!(candidate.walkable);
        assert!(candidate.allied_building);
        assert!(!candidate.standing_tree);
        assert_eq!(frame.points()[index][point_feature::TOKEN_PRESENT], 1.0);
        assert_eq!(frame.points()[index][point_feature::POINTER_VALID], 1.0);
        assert!(view.units.iter().any(|unit| unit.team == tracker.team()
            && PointSource::BuildingLanding(unit.kind) == candidate.source
            && unit.pos.within(candidate.position, Fixed::from_int(600))));
        result.landings += 1;
        for unit in [ControlledUnit::Hero, ControlledUnit::Courier] {
            inspect_landing(side, place, &space, index, candidate, unit);
            if !space.allows(StructuredAction::MovePoint {
                unit,
                point: PointIndex(index),
            }) {
                result.denied.push((unit, index, candidate.source));
            }
        }
    }
    let (index, fountain) = space
        .point_candidates()
        .iter()
        .copied()
        .enumerate()
        .find(|(_, point)| point.source == PointSource::BuildingLanding(UnitKind::Fountain))
        .expect("existing fountain pointer, not a new goal feature");
    (result.arrival, result.fountain_regeneration) =
        fountain_trip(&mut world, side, place, index, fountain.position);
    eprintln!(
        "navigation_summary side={side} place={place:?} landings={} denied={} trees={} closed={} non_tp={} arrival={:?} regeneration={}",
        result.landings,
        result.denied.len(),
        result.trees,
        result.closed,
        result.non_teleport,
        result.arrival,
        result.fountain_regeneration
    );
    result
}

fn encode(tracker: &StateTracker, space: &ActionSpace) -> FeatureFrame {
    assert_eq!(space.tick(), START_TICK);
    let mut encoder = FeatureEncoder::new(tracker);
    encoder.observe(tracker).expect("feature observation");
    let mut frame = FeatureFrame::new();
    encoder
        .encode(
            tracker,
            space,
            &ItemReadiness::new(),
            &LocalPolicyState::new(0),
            &mut frame,
        )
        .expect("seat-safe features without NN inference");
    assert!(frame.is_finite());
    frame
}

fn inspect_landing(
    side: usize,
    place: Place,
    space: &ActionSpace,
    index: usize,
    candidate: PointCandidate,
    unit: ControlledUnit,
) {
    let (mut world, _) = fixture(side, place);
    let named = selector(&world, side, unit);
    let body = world
        .driven_by(world.seats[side].slot, named)
        .expect("live owned body");
    let start = world.transform.get(body).expect("start").pos;
    let command = move_command(&world, side, unit, candidate.position);
    assert!(world.grid.walkable(candidate.position));
    let route = find_path(&world.grid, start, candidate.position);
    assert!(
        world.stats.get(body).expect("body stats").flies
            || grid_los(&world.grid, start, candidate.position)
            || !route.is_empty(),
        "no native route to {candidate:?}"
    );
    assert_eq!(
        world.validate_order(command.slot, command.unit, &command.order),
        Ok(())
    );
    let action = StructuredAction::MovePoint {
        unit,
        point: PointIndex(index),
    };
    let allowed = space.allows(action);
    let body_mask = space.controlled_unit_mask(ActionKind::MovePoint);
    assert!(body_mask.allows(unit));
    assert_move_decoding(space, action, command);
    world.advance(&[command]);
    let end = world.transform.get(body).expect("native first step").pos;
    assert_eq!(
        world.orders.get(body).expect("persistent order").current,
        UnitOrder::Move {
            pos: candidate.position
        }
    );
    eprintln!(
        "navigation_candidate side={side} place={place:?} unit={unit:?} index={index} candidate={candidate:?} pointer_present=1 pointer_valid=1 move_allowed={allowed} attack_move_allowed={} native_validate=Ok native_grid=true route_corners={} start={start:?} first={end:?} first_distance={} initial_distance={} displacement={} arrived={} tp_mask={} native_tp={}",
        space.attack_move_point_mask(unit)[index],
        route.len(),
        distance(end, candidate.position),
        distance(start, candidate.position),
        distance(start, end),
        end == candidate.position,
        space
            .use_target_mask(ControlledUnit::Hero, ItemSlot(0))
            .expect("TP mask")
            .points()[index],
        world.teleport_spot(world.seats[side].team, candidate.position, 600)
    );
}

fn assert_move_decoding(space: &ActionSpace, action: StructuredAction, command: Command) {
    if space.allows(action) {
        let decoded = space
            .decode(action)
            .expect("legal Move decodes")
            .expect("Move order");
        assert_eq!(decoded.unit, command.unit);
        assert_eq!(decoded.order, command.order);
    } else {
        let error = space
            .decode(action)
            .expect_err("masked Move must not decode");
        assert_eq!(error, ActionError::NotAllowed(ActionKind::MovePoint));
        assert_eq!(
            error.to_string(),
            "action MovePoint is masked by the current action space"
        );
    }
}

fn inspect_negative(
    world: &mut World,
    side: usize,
    space: &ActionSpace,
    index: usize,
    candidate: PointCandidate,
    evidence: &mut Evidence,
) {
    let tp = StructuredAction::Use {
        unit: ControlledUnit::Hero,
        slot: ItemSlot(0),
        target: ActionTarget::Point(PointIndex(index)),
    };
    if !candidate.walkable || candidate.standing_tree {
        assert!(!world.grid.walkable(candidate.position));
        for unit in [ControlledUnit::Hero, ControlledUnit::Courier] {
            assert!(!space.move_point_mask(unit)[index]);
            assert!(!space.attack_move_point_mask(unit)[index]);
        }
        assert!(!space.allows(tp));
        assert!(!world.teleport_spot(world.seats[side].team, candidate.position, 600));
        let hero = world.seats[side].unit.expect("hero");
        assert!(!world.begin_teleport(hero, Target::Pos(candidate.position), 90, 600, 0));
        assert!(!world.is_channelling(hero));
        let error = space
            .decode(tp)
            .expect_err("closed/tree TP must remain masked");
        assert_eq!(
            error.to_string(),
            "action Use is masked by the current action space"
        );
        if candidate.standing_tree {
            evidence.trees += 1;
        } else {
            evidence.closed += 1;
        }
    }
    if candidate.walkable && !candidate.allied_building {
        assert!(
            !space.allows(tp),
            "ordinary Move eligibility must not grant TP provenance"
        );
        assert!(space.move_point_mask(ControlledUnit::Hero)[index]);
        assert!(space.move_point_mask(ControlledUnit::Courier)[index]);
        evidence.non_teleport += 1;
    }
    if candidate.source == PointSource::BuildingLanding(UnitKind::Fountain) {
        assert!(
            space.allows(tp),
            "ready TP must still accept its existing fountain landing"
        );
        assert!(world.teleport_spot(world.seats[side].team, candidate.position, 600));
        assert_eq!(
            space
                .controlled_item(ControlledUnit::Hero, ItemSlot(0))
                .expect("scroll")
                .id,
            ItemId(8)
        );
    }
}

fn assert_tp_target_kinds(world: &World, side: usize, space: &ActionSpace) {
    let hero = world.seats[side].unit.expect("hero");
    let mask = space
        .use_target_mask(ControlledUnit::Hero, ItemSlot(0))
        .expect("ready TP mask");
    assert!(!mask.allows_none());
    assert!(mask.entities().iter().all(|allowed| !allowed));
    for target in [Target::None, Target::Unit(wire_id(hero))] {
        assert_eq!(
            world.validate_order(
                world.seats[side].slot,
                None,
                &Order::Use {
                    slot: ItemSlot(0),
                    target
                }
            ),
            Err(RejectReason::WrongTargetKind)
        );
    }
}

fn fountain_trip(
    world: &mut World,
    side: usize,
    place: Place,
    index: usize,
    goal: Vec2,
) -> (Option<u32>, bool) {
    let hero = world.seats[side].unit.expect("hero");
    let start = world.transform.get(hero).expect("start").pos;
    let limit = match place {
        Place::Barracks => SHORT_LIMIT,
        Place::Lane => FOUNTAIN_LIMIT,
    };
    let command = move_command(world, side, ControlledUnit::Hero, goal);
    assert_eq!(
        world.validate_order(command.slot, command.unit, &command.order),
        Ok(())
    );
    assert!(!start.within(
        world.map.fountains[side],
        Fixed::from_int(rules::FOUNTAIN_HEAL_RADIUS)
    ));
    let mut regenerated = false;
    let mut arrival = None;
    for elapsed in 1..=limit {
        let previous_health = world.health.get(hero).expect("health").hp;
        let previous_mana = world.mana.get(hero).expect("mana").mana;
        // Continue decodes to no command: no Teacher prefix or periodic replacement orders.
        world.advance(if elapsed == 1 {
            std::slice::from_ref(&command)
        } else {
            &[]
        });
        assert!(world.alive(hero));
        assert_eq!(
            world.orders.get(hero).expect("held Move").current,
            UnitOrder::Move { pos: goal }
        );
        let position = world.transform.get(hero).expect("position").pos;
        let health = world.health.get(hero).expect("health").hp;
        let mana = world.mana.get(hero).expect("mana").mana;
        let fountain_range = position.within(
            world.map.fountains[side],
            Fixed::from_int(rules::FOUNTAIN_HEAL_RADIUS),
        );
        if fountain_range
            && health - previous_health >= Fixed::from_int(20)
            && mana - previous_mana >= Fixed::from_int(10)
        {
            regenerated = true;
        }
        if [1, 60, 300, 600, 1_200].contains(&elapsed) || position == goal {
            eprintln!(
                "navigation_trip side={side} place={place:?} index={index} elapsed={elapsed} orders_sent=1 position={position:?} goal={goal:?} initial_distance={} remaining={} hp={} mana={} fountain_range={fountain_range} regenerated={regenerated}",
                distance(start, goal),
                distance(position, goal),
                health.to_int(),
                mana.to_int()
            );
        }
        if position == goal {
            arrival = Some(elapsed);
            break;
        }
    }
    (arrival, regenerated)
}

fn move_command(world: &World, side: usize, unit: ControlledUnit, goal: Vec2) -> Command {
    assert!(side < world.seats.len());
    assert!(world.grid.walkable(goal));
    Command {
        slot: world.seats[side].slot,
        unit: selector(world, side, unit),
        order: Order::Move {
            target: Target::Pos(goal),
        },
    }
}

fn selector(world: &World, side: usize, unit: ControlledUnit) -> Option<bota_proto::EntityId> {
    assert!(side < world.seats.len());
    assert!(world.seats[side].unit.is_some());
    match unit {
        ControlledUnit::Hero => None,
        ControlledUnit::Courier => Some(wire_id(world.seats[side].courier.expect("stock courier"))),
    }
}

fn distance(source: Vec2, target: Vec2) -> i64 {
    let squared = source.distance_squared(target);
    assert!(squared >= 0);
    let distance = isqrt64(squared) / i64::from(Fixed::ONE.raw);
    assert!(distance >= 0);
    distance
}
