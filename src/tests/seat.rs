use std::collections::VecDeque;

use bota_proto::{
    EntityId, MapId, MatchInfo, Order, Pick, PlayerId, ServerMsg, SlotId, Team, TickMode, WorldView,
};

#[cfg(feature = "builtin")]
use crate::{ActionKind, MODEL_PARAMETER_COUNT, PolicyModel};
use crate::{SHADOW_FIEND, Seated, Wire, play_idle_on};

struct MockWire {
    messages: VecDeque<ServerMsg>,
    acknowledgements: Vec<u32>,
    orders: Vec<(Option<EntityId>, Order)>,
}

impl Wire for MockWire {
    fn hear(&mut self) -> std::io::Result<Option<ServerMsg>> {
        Ok(self.messages.pop_front())
    }

    fn order(&mut self, unit: Option<EntityId>, order: Order) -> std::io::Result<u32> {
        self.orders.push((unit, order));
        u32::try_from(self.orders.len())
            .map_err(|_| std::io::Error::other("mock order sequence overflow"))
    }

    fn acknowledge(&mut self, tick: u32) -> std::io::Result<()> {
        self.acknowledgements.push(tick);
        Ok(())
    }
}

#[test]
fn seat_loop_acknowledges_first_lockstep_snapshot_and_sends_no_order() {
    let mut wire = mock_wire();
    let outcome = play_idle_on(&mut wire, seated(TickMode::Lockstep), Some(1)).expect("seat plays");

    assert_eq!(outcome.ticks, 1);
    assert_eq!(wire.acknowledgements, [1]);
    assert!(wire.orders.is_empty());
}

#[test]
fn seat_loop_does_not_acknowledge_realtime_snapshot() {
    let mut wire = mock_wire_with_mode(TickMode::Realtime);
    let outcome = play_idle_on(&mut wire, seated(TickMode::Realtime), Some(1)).expect("seat plays");

    assert_eq!(outcome.ticks, 1);
    assert!(wire.acknowledgements.is_empty());
}

#[test]
fn seat_loop_rejects_match_mode_that_differs_from_welcome() {
    let mut wire = mock_wire();

    let error = play_idle_on(&mut wire, seated(TickMode::Realtime), Some(1))
        .expect_err("different match modes must fail");

    assert_eq!(
        error.to_string(),
        "MatchStart mode Lockstep differs from Welcome mode Realtime"
    );
}

#[test]
fn seat_loop_rejects_tick_rate_that_differs_from_welcome() {
    let mut wire = mock_wire();
    let ServerMsg::MatchStart { info } = &mut wire.messages[0] else {
        panic!("first fixture message must start the match");
    };
    info.tick_rate = 60;

    let error = play_idle_on(&mut wire, seated(TickMode::Lockstep), Some(1))
        .expect_err("different tick rates must fail");

    assert_eq!(
        error.to_string(),
        "MatchStart tick rate 60 differs from Welcome tick rate 30"
    );
}

#[test]
fn seat_loop_rejects_snapshot_for_another_team() {
    let mut wire = mock_wire();
    let ServerMsg::Snapshot { view } = &mut wire.messages[1] else {
        panic!("second fixture message must be a snapshot");
    };
    view.viewer = Some(Team::Dire);

    let error = play_idle_on(&mut wire, seated(TickMode::Lockstep), Some(1))
        .expect_err("another team's snapshot must fail");

    assert_eq!(
        error.to_string(),
        "Snapshot viewer Some(Dire) differs from assigned team Some(Radiant)"
    );
}

#[test]
fn seat_loop_rejects_non_shadow_fiend_match_pick() {
    let mut wire = mock_wire();
    let ServerMsg::MatchStart { info } = &mut wire.messages[0] else {
        panic!("first fixture message must start the match");
    };
    info.picks[0].hero = bota_proto::HeroId(1);

    let error = play_idle_on(&mut wire, seated(TickMode::Lockstep), Some(1))
        .expect_err("wrong hero must fail");

    assert_eq!(
        error.to_string(),
        "assigned slot 0 picked HeroId(1), expected Shadow Fiend HeroId(2)"
    );
}

#[cfg(feature = "builtin")]
#[test]
fn deployment_seat_runs_loaded_policy_and_emits_an_order() {
    let (mut arena, start) = crate::Arena::new(crate::ArenaConfig {
        seats: 2,
        map: MapId(1),
        seed: 70_001,
    })
    .expect("arena");
    let next = arena.step(&[None, None]).expect("next arena tick");
    let mut messages = start.messages[0].clone();
    messages.extend(next.messages[0].clone());
    let mut wire = MockWire {
        messages: messages.into(),
        acknowledgements: Vec::new(),
        orders: Vec::new(),
    };
    let model = stop_policy();

    let outcome = crate::play_policy_on(
        &mut wire,
        Seated {
            player: PlayerId(1),
            slot: SlotId(0),
            tick_rate: 30,
            mode: TickMode::Lockstep,
        },
        Some(2),
        &model,
    )
    .expect("policy seat");

    assert_eq!(outcome.decisions, 1);
    assert_eq!(outcome.orders, 1);
    assert_eq!(wire.orders.len(), 1);
}

#[cfg(feature = "builtin")]
#[test]
fn deployment_seat_resends_a_body_order_after_hero_identity_changes() {
    let (_arena, start) = crate::Arena::new(crate::ArenaConfig {
        seats: 2,
        map: MapId(1),
        seed: 70_003,
    })
    .expect("arena");
    let mut messages = start.messages[0].clone();
    let first = messages
        .iter()
        .find_map(|message| match message {
            ServerMsg::Snapshot { view } => Some(view.clone()),
            _ => None,
        })
        .expect("initial snapshot");
    let mut respawned = first.clone();
    respawned.tick = first.tick.checked_add(3).expect("respawn tick");
    replace_owned_hero_generation(&mut respawned);
    let mut finished = respawned.clone();
    finished.tick = respawned.tick.checked_add(1).expect("finish tick");
    messages.push(ServerMsg::Snapshot { view: respawned });
    messages.push(ServerMsg::Snapshot {
        view: finished.clone(),
    });
    let mut wire = MockWire {
        messages: messages.into(),
        acknowledgements: Vec::new(),
        orders: Vec::new(),
    };
    let model = stop_policy();

    let outcome = crate::play_policy_on(
        &mut wire,
        Seated {
            player: PlayerId(1),
            slot: SlotId(0),
            tick_rate: 30,
            mode: TickMode::Lockstep,
        },
        Some(finished.tick),
        &model,
    )
    .expect("policy seat");

    assert_eq!(outcome.decisions, 2);
    assert_eq!(outcome.orders, 2);
    assert_eq!(wire.orders.len(), 2);
}

fn mock_wire() -> MockWire {
    mock_wire_with_mode(TickMode::Lockstep)
}

fn mock_wire_with_mode(mode: TickMode) -> MockWire {
    MockWire {
        messages: VecDeque::from([
            ServerMsg::MatchStart {
                info: match_info(mode),
            },
            ServerMsg::Snapshot {
                view: world_view(1),
            },
        ]),
        acknowledgements: Vec::new(),
        orders: Vec::new(),
    }
}

fn seated(mode: TickMode) -> Seated {
    Seated {
        player: PlayerId(1),
        slot: SlotId(0),
        tick_rate: 30,
        mode,
    }
}

fn match_info(mode: TickMode) -> MatchInfo {
    MatchInfo {
        match_id: 7,
        map: MapId(1),
        tick_rate: 30,
        pregame_ticks: 0,
        trees: Vec::new(),
        terrain_cells: 0,
        terrain_rle: Vec::new(),
        opaque_cells: Vec::new(),
        mode,
        picks: vec![Pick {
            slot: SlotId(0),
            team: Team::Radiant,
            hero: SHADOW_FIEND,
        }],
        shop: Vec::new(),
    }
}

fn world_view(tick: u32) -> WorldView {
    WorldView {
        tick,
        viewer: Some(Team::Radiant),
        units: Vec::new(),
        projectiles: Vec::new(),
        players: Vec::new(),
        felled_trees: Vec::new(),
        planted_trees: Vec::new(),
        loot: Vec::new(),
    }
}

#[cfg(feature = "builtin")]
fn replace_owned_hero_generation(view: &mut WorldView) {
    let player = view
        .players
        .iter_mut()
        .find(|player| player.slot == SlotId(0))
        .expect("owned player");
    let previous = player.unit.expect("owned hero");
    let current = EntityId {
        idx: previous.idx,
        generation: previous.generation.checked_add(1).expect("hero generation"),
    };
    player.unit = Some(current);
    let hero = view
        .units
        .iter_mut()
        .find(|unit| unit.id == previous)
        .expect("owned hero unit");
    hero.id = current;
}

#[cfg(feature = "builtin")]
fn stop_policy() -> PolicyModel {
    let model = PolicyModel::fresh(70_002).expect("model");
    let mut parameters = vec![0.0; MODEL_PARAMETER_COUNT];
    let mut offset = 0usize;
    for (name, shape) in model.parameter_schema().expect("parameter schema") {
        if name == "kind.bias" {
            parameters[offset + ActionKind::Stop.index()] = 10.0;
            model.import_parameters(&parameters).expect("stop policy");
            return model;
        }
        offset += shape.iter().product::<usize>();
    }
    panic!("kind bias parameter is missing");
}
