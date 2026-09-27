#[cfg(feature = "builtin")]
use bota_proto::{EntityId, Order};
use bota_proto::{MapId, MatchInfo, PlayerId, ServerMsg, SlotId, Team, TickMode, WorldView};

use super::readiness;
use crate::tests::support::RecordingWire as MockWire;
#[cfg(feature = "builtin")]
use crate::{ActionKind, PolicyModel};
use crate::{Seated, play_idle_on};

pub(super) fn tactical_directory(name: &str) -> std::path::PathBuf {
    let directory = std::env::temp_dir().join(format!(
        "drysua-tactical-live-{}-{name}",
        std::process::id()
    ));
    std::fs::create_dir(&directory).expect("unique test directory");
    directory
}

#[cfg(feature = "builtin")]
fn tactical_combat_fixture(map: MapId) -> (MatchInfo, WorldView) {
    let (_, messages) = super::neural_order_contract::combat_start(map, 70_007, 0, None);
    let [
        ServerMsg::MatchStart { info },
        ServerMsg::Snapshot { view },
        ..,
    ] = &messages[..]
    else {
        panic!("combat projection");
    };
    let mut info = info.clone();
    let mut view = view.clone();
    info.pregame_ticks = 0;
    info.terrain_cells = 128;
    info.terrain_rle = vec![(16384, 0x80)];
    info.trees.clear();
    info.opaque_cells.clear();
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
fn idle_seat_acknowledges_only_lockstep_and_rejects_invalid_assignments_before_output() {
    for mode in [TickMode::Lockstep, TickMode::Realtime] {
        let mut wire = mock_wire_with_mode(mode);
        let outcome = play_idle_on(&mut wire, seated(mode), Some(1)).expect("idle seat plays");
        assert_eq!(outcome.ticks, 1);
        assert_eq!(wire.acknowledgements, expected_acknowledgements(mode, 1));
        assert!(wire.orders.is_empty());
    }
    for (scenario, expected) in [
        (
            0,
            "MatchStart mode Lockstep differs from Welcome mode Realtime",
        ),
        (
            1,
            "MatchStart tick rate 60 differs from Welcome tick rate 30",
        ),
        (
            2,
            "Snapshot viewer Some(Dire) differs from assigned team Some(Radiant)",
        ),
        (
            3,
            "assigned slot 0 picked HeroId(1), expected Shadow Fiend HeroId(2)",
        ),
    ] {
        let mut wire = mock_wire();
        let ServerMsg::MatchStart { info } = &mut wire.messages[0] else {
            panic!("match start");
        };
        match scenario {
            1 => info.tick_rate = 60,
            3 => info.picks[0].hero = bota_proto::HeroId(1),
            _ => {}
        }
        if scenario == 2 {
            let ServerMsg::Snapshot { view } = &mut wire.messages[1] else {
                panic!("snapshot");
            };
            view.viewer = Some(Team::Dire);
        }
        let mode = if scenario == 0 {
            TickMode::Realtime
        } else {
            TickMode::Lockstep
        };
        let error = play_idle_on(&mut wire, seated(mode), Some(1)).expect_err(expected);
        assert_eq!(error.to_string(), expected);
        assert!(wire.orders.is_empty());
        assert!(wire.acknowledgements.is_empty());
    }
}

#[test]
fn seat_loop_rejects_an_unbounded_message_stream_without_snapshots() {
    let mut wire = recording_wire(std::iter::repeat_n(
        ServerMsg::LobbyState { slots: Vec::new() },
        4_097,
    ));

    let error = play_idle_on(&mut wire, seated(TickMode::Lockstep), None)
        .expect_err("message stream must make snapshot progress");

    assert_eq!(
        error.to_string(),
        "server sent too many messages without a snapshot"
    );
}

#[test]
fn all_controllers_require_complete_ticks_with_or_without_a_live_hero() {
    let model = crate::PolicyModel::fresh(70_011).expect("model");
    for (map, hero_present) in [
        (MapId(0), false),
        (MapId(1), false),
        (MapId(0), true),
        (MapId(1), true),
    ] {
        for controller in 0..3 {
            let mut wire = mock_wire();
            let ServerMsg::MatchStart { info } = &mut wire.messages[0] else {
                panic!("match start");
            };
            info.map = map;
            let mut view = world_view(1);
            if !hero_present {
                let hero = view.players[0].unit.take().expect("hero");
                view.units.retain(|unit| unit.id != hero);
            }
            wire.messages[1] = ServerMsg::Snapshot { view: view.clone() };
            view.tick = 2;
            wire.messages.push_back(ServerMsg::Snapshot { view });
            let seat = seated(TickMode::Lockstep);
            let error = match controller {
                0 => crate::play_teacher_on(&mut wire, seat, None),
                1 => crate::play_policy_on(&mut wire, seat, None, &model),
                _ => crate::play_neural_on(&mut wire, seat, None, &model),
            }
            .expect_err("tick completion is mandatory");
            assert_eq!(
                error.to_string(),
                "server sent Snapshot before completing the previous tick"
            );
            assert!(wire.orders.is_empty());
            assert!(wire.acknowledgements.is_empty());
        }
    }
}

#[cfg(feature = "builtin")]
#[test]
fn teacher_tcp_cli_plays_map_one_without_weights() {
    tcp_cli_match(MapId(1), 70_006, "teacher", 1_000, None);
}

#[cfg(feature = "builtin")]
#[test]
fn controllers_keep_cadence_and_teacher_learning_but_neural_continue_never_buys_or_learns() {
    let stop = stop_policy();
    let continue_policy = biased_policy(ActionKind::Continue);
    for (map, limit) in [(MapId(0), 8), (MapId(1), 8), (MapId(1), 2)] {
        let messages = arena_messages(map, 70_009, limit);
        for mode in [TickMode::Lockstep, TickMode::Realtime] {
            for controller in 0..3 {
                let mut wire = recording_wire(messages.clone());
                let ServerMsg::MatchStart { info } = &mut wire.messages[0] else {
                    panic!("MatchStart first");
                };
                if map == MapId(1) {
                    assert_eq!(info.pregame_ticks, 900);
                }
                info.mode = mode;
                if limit == 2 {
                    info.pregame_ticks = 0;
                }
                let outcome = match controller {
                    0 => crate::play_teacher_on(&mut wire, seated(mode), Some(limit)),
                    1 => crate::play_policy_on(&mut wire, seated(mode), Some(limit), &stop),
                    _ => crate::play_neural_on(
                        &mut wire,
                        seated(mode),
                        Some(limit),
                        &continue_policy,
                    ),
                }
                .expect("pregame decisions");
                assert_eq!(outcome.decisions, (limit - 2) / 3 + 1, "ticks 1, 4, 7");
                assert_eq!(wire.orders.is_empty(), controller == 2);
                assert_eq!(outcome.orders == 0, controller == 2);
                assert_eq!(
                    wire.acknowledgements,
                    expected_acknowledgements(mode, limit)
                );
                if limit == 2 && controller != 2 {
                    assert_eq!(wire.orders.len(), 1);
                    assert!(match wire.orders[0].1 {
                        Order::Learn { .. } => controller == 0,
                        Order::Move { .. } => controller == 1,
                        _ => false,
                    });
                }
            }
        }
    }
}

fn mock_wire() -> MockWire {
    mock_wire_with_mode(TickMode::Lockstep)
}

fn recording_wire(messages: impl IntoIterator<Item = ServerMsg>) -> MockWire {
    MockWire {
        messages: messages.into_iter().collect(),
        acknowledgements: Vec::new(),
        orders: Vec::new(),
    }
}

fn expected_acknowledgements(mode: TickMode, ticks: u32) -> Vec<u32> {
    if mode == TickMode::Lockstep {
        (1..=ticks).collect()
    } else {
        Vec::new()
    }
}

#[cfg(feature = "builtin")]
fn arena_messages(map: MapId, seed: u64, ticks: u32) -> Vec<ServerMsg> {
    assert!((1..=8).contains(&ticks));
    let (mut arena, start) = crate::Arena::new(crate::ArenaConfig {
        seats: 2,
        map,
        seed,
    })
    .expect("native arena");
    let mut messages = start.messages[0].clone();
    for _ in 1..ticks {
        messages.extend(arena.step(&[None; 2]).expect("tick").messages[0].clone());
    }
    assert!(matches!(
        messages.first(),
        Some(ServerMsg::MatchStart { .. })
    ));
    messages
}

fn mock_wire_with_mode(mode: TickMode) -> MockWire {
    recording_wire([
        ServerMsg::MatchStart {
            info: match_info(mode),
        },
        ServerMsg::Snapshot {
            view: world_view(1),
        },
    ])
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
    let mut info = readiness::match_info();
    info.mode = mode;
    info
}

fn world_view(tick: u32) -> WorldView {
    readiness::world_view(tick, &[None; 9], &[None; 6])
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
    biased_policy(ActionKind::Stop)
}

#[cfg(feature = "builtin")]
fn biased_policy(kind: ActionKind) -> PolicyModel {
    use super::neural_order_contract::{DiagnosticPolicy, install_policy};
    let model = PolicyModel::fresh(70_002).expect("model");
    install_policy(&model, DiagnosticPolicy::Constant(kind));
    model
}

#[cfg(feature = "builtin")]
#[test]
fn neural_sends_network_stop_instead_of_teacher_on_both_maps_at_low_health_and_during_channels() {
    for map in [MapId(0), MapId(1)] {
        for scenario in 0..3 {
            let (info, mut view) = tactical_combat_fixture(map);
            if scenario == 1 {
                view.units[0].hp = 1;
            }
            if scenario == 2 {
                view.units[0].statuses.bits = bota_proto::StatusFlags::CHANNELLING;
            }
            let mut tracker = crate::StateTracker::new(SlotId(0), &info).expect("tracker");
            tracker.observe_snapshot(&view).expect("snapshot");
            let (baseline, _) = crate::Teacher::new()
                .decide(
                    &tracker,
                    &crate::OrderPersistence::default(),
                    &crate::ItemReadiness::new(),
                )
                .expect("teacher baseline");
            assert_ne!(baseline.kind(), ActionKind::Stop);
            let mut finished = view.clone();
            finished.tick += 1;
            let mut wire = recording_wire([
                ServerMsg::MatchStart { info },
                ServerMsg::Snapshot { view: view.clone() },
                ServerMsg::Events {
                    tick: view.tick,
                    events: Vec::new(),
                },
                ServerMsg::Snapshot { view: finished },
            ]);
            let outcome = crate::seat::play_neural_on(
                &mut wire,
                seated(TickMode::Lockstep),
                Some(view.tick + 1),
                &stop_policy(),
            )
            .expect("pure neural");
            assert_eq!(outcome.decisions, 1);
            assert_eq!(wire.orders.len(), 1);
            assert_eq!(
                wire.orders[0],
                (
                    None,
                    Order::Move {
                        target: bota_proto::Target::None
                    }
                )
            );
        }
    }
}

#[cfg(feature = "builtin")]
#[test]
fn neural_tcp_cli_loads_current_seed_weights_on_both_maps_without_training() {
    let directory = tactical_directory("neural-tcp");
    let model = PolicyModel::fresh(70_010).expect("untrained seed model");
    crate::TrainingArtifact::save_runtime_weights(&model, &directory).expect("current metadata");
    for map in [MapId(0), MapId(1)] {
        tcp_cli_match(map, 70_010, "neural", 32, Some(&directory));
    }
    std::fs::remove_dir_all(directory).expect("remove seed artifact");
}

#[cfg(feature = "builtin")]
fn tcp_cli_match(
    map: MapId,
    seed: u64,
    policy: &str,
    limit: u32,
    directory: Option<&std::path::Path>,
) {
    use std::{net::TcpListener, sync::mpsc, thread, time::Duration};
    let listener = TcpListener::bind("127.0.0.1:0").expect("real TCP");
    let address = listener.local_addr().expect("address").to_string();
    let (finished, completion) = mpsc::sync_channel(1);
    thread::spawn(move || {
        let result = bota_server::game_loop::run(
            listener,
            bota_server::game_loop::ServerOpts {
                mode: TickMode::Lockstep,
                tick_rate: 30,
                players: 2,
                replay: None,
                seed,
                map,
                ack_timeout_ticks: 150,
                cheats: false,
            },
        );
        finished.send(result).expect("completion receiver");
    });
    let (mut idle, idle_seat) =
        crate::Link::join_with_timeout(&address, "idle", Duration::from_secs(5)).expect("opponent");
    assert_eq!(idle_seat.slot, SlotId(0));
    let opponent = thread::spawn(move || play_idle_on(&mut idle, idle_seat, Some(limit + 10)));
    let limit_argument = limit.to_string();
    let mut arguments = vec![
        "drysua",
        "play",
        "--policy",
        policy,
        "--addr",
        &address,
        "--limit",
        &limit_argument,
    ];
    if let Some(directory) = directory {
        arguments.extend(["--weights-directory", directory.to_str().expect("path")]);
    }
    crate::cli::run_from_for_test(arguments).expect("real TCP policy CLI");
    let outcome = opponent
        .join()
        .expect("opponent thread")
        .expect("opponent result");
    assert!(outcome.ticks >= limit);
    assert_eq!(outcome.orders, 0);
    completion
        .recv_timeout(Duration::from_secs(5))
        .expect("bounded exit")
        .expect("server result");
}

#[cfg(feature = "builtin")]
#[test]
fn deployment_and_neural_deduplicate_body_orders_but_resend_after_rejection_or_respawn() {
    for (map, neural, reject, respawn_tick, limit) in [
        (MapId(1), false, false, 4, 5),
        (MapId(0), true, false, 7, 11),
        (MapId(0), true, true, 7, 11),
        (MapId(1), true, false, 7, 11),
        (MapId(1), true, true, 7, 11),
    ] {
        let (info, mut view) = if neural {
            tactical_combat_fixture(map)
        } else {
            deployment_stop_fixture()
        };
        let mut messages = vec![ServerMsg::MatchStart { info }];
        let ticks = if neural {
            (1..=limit).collect::<Vec<_>>()
        } else {
            vec![1, 4, 5]
        };
        for tick in ticks.iter().copied() {
            view.tick = tick;
            if tick == respawn_tick {
                replace_owned_hero_generation(&mut view);
            }
            messages.push(ServerMsg::Snapshot { view: view.clone() });
            messages.push(ServerMsg::Events {
                tick,
                events: Vec::new(),
            });
            if tick == 1 && reject {
                messages.push(ServerMsg::OrderRejected {
                    seq: 1,
                    reason: bota_proto::RejectReason::UnknownTarget,
                });
            }
        }
        let mut wire = recording_wire(messages);
        let model = stop_policy();
        let seat = seated(TickMode::Lockstep);
        let outcome = if neural {
            crate::play_neural_on(&mut wire, seat, Some(limit), &model)
        } else {
            crate::play_policy_on(&mut wire, seat, Some(limit), &model)
        }
        .expect("tracked deployment");
        assert_eq!(outcome.decisions, (limit - 2) / 3 + 1);
        assert_eq!(outcome.orders, if reject { 3 } else { 2 });
        assert_eq!(wire.orders.len(), outcome.orders as usize);
        assert_eq!(outcome.rejections, u32::from(reject));
        assert_eq!(wire.acknowledgements, ticks);
        assert!(wire.orders.iter().all(|order| *order
            == (
                None,
                Order::Move {
                    target: bota_proto::Target::None
                }
            )));
    }
}

#[cfg(feature = "builtin")]
fn deployment_stop_fixture() -> (MatchInfo, WorldView) {
    // Map0 deployment delegates to Teacher; only Map1 uses the forced Stop model.
    let (_, start) = crate::Arena::new(crate::ArenaConfig {
        seats: 2,
        map: MapId(1),
        seed: 70_003,
    })
    .expect("deployment arena");
    let mut info = start.messages[0]
        .iter()
        .find_map(|message| match message {
            ServerMsg::MatchStart { info } => Some(info.clone()),
            _ => None,
        })
        .expect("deployment match info");
    info.pregame_ticks = 0;
    let view = start.messages[0]
        .iter()
        .find_map(|message| match message {
            ServerMsg::Snapshot { view } => Some(view.clone()),
            _ => None,
        })
        .expect("deployment snapshot");
    assert_eq!(info.map, MapId(1));
    assert_eq!(view.tick, 1);
    (info, view)
}

#[test]
fn neural_rejects_missing_weights_before_attempting_connection() {
    let directory = tactical_directory("neural-missing");
    let model = crate::PolicyModel::fresh(0).expect("model");
    let expected = crate::TrainingArtifact::load_runtime_weights(&model, &directory)
        .expect_err("missing weights")
        .to_string();
    let error = crate::play_neural("invalid address", "neural", None, &directory)
        .expect_err("no fallback or connection");
    assert_eq!(error.to_string(), expected);
    assert_eq!(error.kind(), std::io::ErrorKind::Other);
    std::fs::remove_dir_all(directory).expect("remove fixture");
}
