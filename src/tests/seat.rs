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
fn tactical_loader_rejects_wrong_length_schema_and_nonfinite_parameters() {
    let directory = tactical_directory("invalid");
    let valid = crate::TacticalPolicy::default().to_bytes();
    let mut schema = valid.clone();
    schema[0] ^= 1;
    let mut nonfinite = valid.clone();
    let offset = crate::TACTICAL_SCHEMA_DESCRIPTOR.len();
    nonfinite[offset..offset + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    let mut oversized = valid.clone();
    oversized.extend_from_slice(&[0; 32]);
    for (bytes, message) in [
        (
            valid[..valid.len() - 1].to_vec(),
            format!(
                "tactical file has {} bytes; expected {}",
                valid.len() - 1,
                valid.len()
            ),
        ),
        (
            oversized,
            format!(
                "tactical file has {} bytes; expected {}",
                valid.len() + 1,
                valid.len()
            ),
        ),
        (
            schema,
            "tactical file schema does not match this architecture and feature order".to_owned(),
        ),
        (
            nonfinite,
            "tactical parameter 0 must be finite and in [-4, 4]".to_owned(),
        ),
    ] {
        std::fs::write(directory.join("drysua.tactical.bin"), bytes).expect("invalid weights");
        let error = crate::seat::play_tactical("unused", "", None, &directory)
            .expect_err("invalid artifact cannot connect");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().ends_with(&message), "{error}");
    }
    std::fs::remove_dir_all(directory).expect("remove weights");
}

#[cfg(unix)]
#[test]
fn tactical_loader_rejects_symlink_instead_of_loading_another_artifact() {
    let directory = tactical_directory("symlink");
    std::os::unix::fs::symlink("missing", directory.join("drysua.tactical.bin")).expect("symlink");
    let error = crate::seat::play_tactical("unused", "", None, &directory).expect_err("no links");
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    assert!(
        error
            .to_string()
            .ends_with("expected a regular non-symlink file")
    );
    std::fs::remove_dir_all(directory).expect("remove link");
}

pub(super) fn tactical_directory(name: &str) -> std::path::PathBuf {
    let directory = std::env::temp_dir().join(format!(
        "drysua-tactical-live-{}-{name}",
        std::process::id()
    ));
    std::fs::create_dir(&directory).expect("unique test directory");
    directory
}

#[cfg(feature = "builtin")]
#[test]
fn tactical_seat_matches_direct_policy_orders_and_default_teacher_on_both_maps() {
    for map in [MapId(0), MapId(1)] {
        let (info, view) = tactical_combat_fixture(map);
        let mut parameters = [0.0; crate::TACTICAL_PARAMETERS];
        parameters[crate::TACTICAL_OUTPUT_BIAS_OFFSET + crate::TacticalMode::Fight.index()] = 1.0;
        let fight = crate::TacticalPolicy::from_parameters(&parameters).expect("fight policy");
        let mut tracker = crate::StateTracker::new(SlotId(0), &info).expect("tracker");
        tracker.observe_snapshot(&view).expect("snapshot");
        let (baseline, _) = crate::Teacher::new()
            .decide(
                &tracker,
                &crate::OrderPersistence::default(),
                &crate::ItemReadiness::new(),
            )
            .expect("baseline");
        for policy in [crate::TacticalPolicy::default(), fight] {
            let (action, space) = crate::Teacher::new()
                .decide_tactical(
                    &tracker,
                    &crate::OrderPersistence::default(),
                    &crate::ItemReadiness::new(),
                    &policy,
                )
                .expect("direct tactical");
            assert_eq!(
                action == baseline,
                policy == crate::TacticalPolicy::default()
            );
            let expected: Vec<_> = space
                .decode(action)
                .expect("decode")
                .into_iter()
                .map(|issued| (issued.unit, issued.order))
                .collect();
            let mut finished = view.clone();
            finished.tick += 1;
            let mut wire = MockWire {
                messages: VecDeque::from([
                    ServerMsg::MatchStart { info: info.clone() },
                    ServerMsg::Snapshot { view: view.clone() },
                    ServerMsg::Events {
                        tick: view.tick,
                        events: Vec::new(),
                    },
                    ServerMsg::Snapshot { view: finished },
                ]),
                acknowledgements: Vec::new(),
                orders: Vec::new(),
            };
            let outcome = crate::seat::play_tactical_on(
                &mut wire,
                seated(TickMode::Lockstep),
                Some(view.tick + 1),
                &policy,
            )
            .expect("tactical live state machine");
            assert_eq!(wire.orders, expected);
            assert_eq!(outcome.decisions, 1);
            assert_eq!(wire.acknowledgements, [view.tick, view.tick + 1]);
        }
    }
}

#[cfg(feature = "builtin")]
fn tactical_combat_fixture(map: MapId) -> (MatchInfo, WorldView) {
    let (_, start) = crate::Arena::new(crate::ArenaConfig {
        seats: 2,
        map,
        seed: 70_007,
    })
    .expect("arena");
    let mut info = start.messages[0]
        .iter()
        .find_map(|message| match message {
            ServerMsg::MatchStart { info } => Some(info.clone()),
            _ => None,
        })
        .expect("match info");
    info.pregame_ticks = 0;
    info.terrain_cells = 128;
    info.terrain_rle = vec![(16384, 0x80)];
    info.trees.clear();
    info.opaque_cells.clear();
    let mut view = start.messages[0]
        .iter()
        .find_map(|message| match message {
            ServerMsg::Snapshot { view } => Some(view.clone()),
            _ => None,
        })
        .expect("snapshot");
    let own = view.players[0].unit.expect("hero identity");
    let mut hero = view
        .units
        .iter()
        .find(|unit| unit.id == own)
        .expect("hero")
        .clone();
    hero.pos = bota_proto::Vec2::from_ints(3000, 3000);
    hero.hp = hero.max_hp;
    hero.mana = hero.max_mana;
    for ability in &mut hero.abilities {
        ability.can_level = false;
    }
    let mut enemy = hero.clone();
    enemy.id = EntityId {
        idx: own.idx + 1000,
        generation: 1,
    };
    enemy.team = Team::Dire;
    enemy.owner = Some(SlotId(1));
    enemy.pos = bota_proto::Vec2::from_ints(3800, 3000);
    enemy.hp = 1;
    view.players[0].gold = Some(0);
    view.players[1].unit = Some(enemy.id);
    view.units = vec![hero, enemy];
    (info, view)
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

#[test]
fn seat_loop_rejects_an_unbounded_message_stream_without_snapshots() {
    let mut wire = MockWire {
        messages: std::iter::repeat_n(ServerMsg::LobbyState { slots: Vec::new() }, 4_097).collect(),
        acknowledgements: Vec::new(),
        orders: Vec::new(),
    };

    let error = play_idle_on(&mut wire, seated(TickMode::Lockstep), None)
        .expect_err("message stream must make snapshot progress");

    assert_eq!(
        error.to_string(),
        "server sent too many messages without a snapshot"
    );
}

#[cfg(feature = "builtin")]
#[test]
fn teacher_seat_on_map_one_learns_without_a_model_and_acknowledges_ticks() {
    let (mut arena, start) = crate::Arena::new(crate::ArenaConfig {
        seats: 2,
        map: MapId(1),
        seed: 70_004,
    })
    .expect("arena");
    let next = arena.step(&[None, None]).expect("next tick");
    let mut messages = start.messages[0].clone();
    messages.extend(next.messages[0].clone());
    set_pregame_ticks(&mut messages, 0);
    let mut wire = MockWire {
        messages: messages.into(),
        acknowledgements: Vec::new(),
        orders: Vec::new(),
    };

    let outcome = crate::play_teacher_on(&mut wire, seated(TickMode::Lockstep), Some(2))
        .expect("model-free Teacher seat");

    assert_eq!(outcome.decisions, 1);
    assert_eq!(outcome.orders, 1);
    assert!(matches!(wire.orders[0].1, Order::Learn { .. }));
    assert_eq!(wire.acknowledgements, [1, 2]);
}

#[cfg(feature = "builtin")]
#[test]
fn teacher_seat_rejects_incomplete_snapshot_ticks() {
    let (mut arena, start) = crate::Arena::new(crate::ArenaConfig {
        seats: 2,
        map: MapId(1),
        seed: 70_005,
    })
    .expect("arena");
    let next = arena.step(&[None, None]).expect("next tick");
    let mut messages = start.messages[0].clone();
    messages.extend(next.messages[0].clone());
    messages.retain(|message| !matches!(message, ServerMsg::Events { .. }));
    let mut wire = MockWire {
        messages: messages.into(),
        acknowledgements: Vec::new(),
        orders: Vec::new(),
    };

    let error = crate::play_teacher_on(&mut wire, seated(TickMode::Lockstep), None)
        .expect_err("tick completion is mandatory for Teacher too");

    assert_eq!(
        error.to_string(),
        "server sent Snapshot before completing the previous tick"
    );
    assert!(wire.orders.is_empty());
    assert!(wire.acknowledgements.is_empty());
}

#[cfg(feature = "builtin")]
#[test]
fn teacher_tcp_cli_plays_map_one_without_weights() {
    use std::{net::TcpListener, sync::mpsc, thread, time::Duration};

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind real server");
    let address = listener.local_addr().expect("server address").to_string();
    let (finished, completion) = mpsc::sync_channel(1);
    thread::spawn(move || {
        let result = bota_server::game_loop::run(
            listener,
            bota_server::game_loop::ServerOpts {
                mode: TickMode::Lockstep,
                tick_rate: 30,
                players: 2,
                replay: None,
                seed: 70_006,
                map: MapId(1),
                ack_timeout_ticks: 150,
            },
        );
        finished.send(result).expect("server completion receiver");
    });
    let (mut idle, idle_seat) =
        crate::Link::join_with_timeout(&address, "idle", Duration::from_secs(5))
            .expect("idle Radiant seat");
    assert_eq!(idle_seat.slot, SlotId(0));
    let opponent = thread::spawn(move || play_idle_on(&mut idle, idle_seat, Some(1_010)));

    crate::cli::run_from_for_test([
        "drysua",
        "play",
        "--policy",
        "teacher",
        "--addr",
        &address,
        "--limit",
        "1000",
        "--weights-directory",
        "artifacts/temp/nonexistent-teacher-weights",
    ])
    .expect("weights-free CLI plays real Map1 TCP match");

    let opponent = opponent
        .join()
        .expect("opponent thread")
        .expect("opponent result");
    assert!(opponent.ticks >= 1_000);
    assert_eq!(opponent.orders, 0);
    completion
        .recv_timeout(Duration::from_secs(5))
        .expect("server exits within bound")
        .expect("server result");
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
    set_pregame_ticks(&mut messages, 0);
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
    assert!(matches!(wire.orders[0].1, Order::Move { .. }));
}

#[cfg(feature = "builtin")]
#[test]
fn pregame_deployment_decides_every_three_ticks_and_only_acknowledges_lockstep() {
    let (mut arena, start) = crate::Arena::new(crate::ArenaConfig {
        seats: 2,
        map: MapId(1),
        seed: 70_008,
    })
    .expect("real pregame MatchStart");
    let mut messages = start.messages[0].clone();
    for _ in 1..8 {
        messages.extend(arena.step(&[None; 2]).expect("tick").messages[0].clone());
    }
    let model = stop_policy();
    let tactical = crate::TacticalPolicy::default();
    for mode in [TickMode::Lockstep, TickMode::Realtime] {
        for controller in 0..3 {
            let mut wire = MockWire {
                messages: messages.clone().into(),
                acknowledgements: Vec::new(),
                orders: Vec::new(),
            };
            let ServerMsg::MatchStart { info } = &mut wire.messages[0] else {
                panic!("MatchStart first");
            };
            assert_eq!(info.pregame_ticks, 900);
            info.mode = mode;
            let outcome = match controller {
                0 => crate::play_teacher_on(&mut wire, seated(mode), Some(8)),
                1 => crate::play_tactical_on(&mut wire, seated(mode), Some(8), &tactical),
                _ => crate::play_policy_on(&mut wire, seated(mode), Some(8), &model),
            }
            .expect("pregame decisions");
            assert_eq!(outcome.decisions, 3, "ticks 1, 4, 7");
            assert!(outcome.orders > 0);
            let expected = if mode == TickMode::Lockstep {
                (1..=8).collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            assert_eq!(wire.acknowledgements, expected);
        }
    }
}

#[cfg(feature = "builtin")]
#[test]
fn deployment_rejects_a_new_snapshot_before_the_tick_complete_events() {
    let (mut arena, start) = crate::Arena::new(crate::ArenaConfig {
        seats: 2,
        map: MapId(1),
        seed: 70_002,
    })
    .expect("arena");
    let next = arena.step(&[None, None]).expect("next arena tick");
    let mut messages = start.messages[0]
        .iter()
        .filter(|message| !matches!(message, ServerMsg::Events { .. }))
        .cloned()
        .collect::<Vec<_>>();
    messages.extend(
        next.messages[0]
            .iter()
            .filter(|message| matches!(message, ServerMsg::Snapshot { .. }))
            .cloned(),
    );
    set_pregame_ticks(&mut messages, 0);
    let mut wire = MockWire {
        messages: messages.into(),
        acknowledgements: Vec::new(),
        orders: Vec::new(),
    };

    let error = crate::play_policy_on(&mut wire, seated(TickMode::Lockstep), None, &stop_policy())
        .expect_err("tick completion is mandatory");

    assert_eq!(
        error.to_string(),
        "server sent Snapshot before completing the previous tick"
    );
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
    set_pregame_ticks(&mut messages, 0);
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
    let respawned_tick = respawned.tick;
    messages.push(ServerMsg::Snapshot { view: respawned });
    messages.push(ServerMsg::Events {
        tick: respawned_tick,
        events: Vec::new(),
    });
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

#[cfg(feature = "builtin")]
fn set_pregame_ticks(messages: &mut [ServerMsg], pregame_ticks: u32) {
    let info = messages
        .iter_mut()
        .find_map(|message| match message {
            ServerMsg::MatchStart { info } => Some(info),
            _ => None,
        })
        .expect("arena MatchStart");
    info.pregame_ticks = pregame_ticks;
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
