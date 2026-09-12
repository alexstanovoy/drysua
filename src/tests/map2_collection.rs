use super::*;
use crate::Map2RewardEnd;
use bota_proto::{DamageKind, EntityId, Vec2};

#[test]
fn map2_collection_defaults_and_budget_match_the_native_cap() {
    let settings = crate::cli::training_settings_for_test(&[]).expect("Map2 defaults");
    assert_eq!(settings.map, MapId(2));
    assert_eq!(PpoSmokeConfig::default().map, MapId(2));
    assert_eq!(LeagueSmokeConfig::default().map, MapId(2));
    assert_eq!(TICK_CAP, bota_server::game::MAP2_TICK_CAP);
    assert_eq!(ACTOR_DECISIONS, 9_300);
    assert_eq!(RETAINED_PER_EPISODE, 1_163);
    assert_eq!(settings.ppo.gamma_tick, 1.0);
    validate_training_job(&settings, false).expect("production config");
    for map in [MapId(0), MapId(1)] {
        let mut invalid = settings.clone();
        invalid.map = map;
        assert_eq!(
            validate_training_job(&invalid, false)
                .expect_err("historical map")
                .to_string(),
            "invalid PPO config field: production training requires Map2"
        );
    }
}

#[test]
fn map2_neutral_is_a_draw_for_candidate_opponent_and_evaluation() {
    for side in 0..2 {
        let environment = fixture_environment(1, side, OpponentSpec::Weak);
        assert_eq!(
            terminal_outcome(&environment, Some(Team::Neutral)),
            Some(PpoTerminalOutcome::Draw)
        );
        assert_eq!(
            evaluation_result(&environment, Some(Team::Neutral)).expect("draw"),
            LeagueMatchResult::Draw
        );
        assert_eq!(
            checkpoint_evaluation_outcome(
                environment.seats[side].tracker.team(),
                Some(Team::Neutral)
            ),
            CheckpointEvaluationOutcome::Draw
        );
        assert_eq!(terminal_outcome(&environment, None), None);
    }
}

#[test]
fn map2_candidate_and_policy_opponent_consume_every_complete_tick_before_retention() {
    let model = Arc::new(PolicyModel::fresh(982_002).expect("opponent"));
    for side in 0..2 {
        let mut environment =
            fixture_environment(1, side, OpponentSpec::SharedPolicy(Arc::clone(&model)));
        let snapshots: Vec<_> = environment
            .seats
            .iter()
            .map(|seat| seat.tracker.current().expect("view").clone())
            .collect();
        let heroes: Vec<_> = snapshots[0]
            .players
            .iter()
            .map(|row| row.unit.expect("hero"))
            .collect();
        for tick in 2..=4 {
            for (index, seat) in environment.seats.iter_mut().enumerate() {
                let mut view = snapshots[index].clone();
                view.tick = tick;
                let events = vec![damage(heroes[index], heroes[1 - index], 1); 100];
                assert!(
                    observe_messages(
                        seat,
                        &[
                            ServerMsg::Snapshot { view },
                            ServerMsg::Events { tick, events }
                        ]
                    )
                    .expect("ordered full tick")
                    .is_none()
                );
                assert_eq!(
                    seat.tracker
                        .map2_reward_state()
                        .expect("reward state")
                        .completed_tick,
                    Some(tick)
                );
            }
        }
        for seat in &mut environment.seats {
            let result = seat
                .tracker
                .take_map2_reward_interval()
                .expect("all events");
            assert_eq!(result.ticks, 3);
            assert_eq!(result.observations.hero_damage_dealt, 300);
            assert_eq!(
                seat.tracker
                    .take_map2_reward_interval()
                    .expect("no duplicate")
                    .ticks,
                0
            );
        }
    }
}

#[test]
fn map2_collection_rejects_match_over_before_final_events_without_consuming_the_tick() {
    let mut environment = fixture_environment(TICK_CAP - 1, 0, OpponentSpec::Weak);
    let mut messages = environment
        .arena
        .step(&[None, None])
        .expect("native final tick")
        .messages
        .remove(0);
    messages.swap(1, 2);
    let seat = &mut environment.seats[0];
    let before = seat.tracker.map2_reward_state();
    let error = observe_messages(seat, &messages).expect_err("reordered terminal");
    assert_eq!(
        error.to_string(),
        "invalid PPO transition: arena Snapshot/Events/MatchOver ordering"
    );
    assert_eq!(seat.tracker.map2_reward_state(), before);
}

#[test]
fn map2_draw_exactly_at_a_full_retention_boundary_is_one_terminal_sample_per_seat() {
    let model = stop_model();
    let settings = crate::cli::training_settings_for_test(&[
        "--environments",
        "2",
        "--rollout",
        &crate::MAP2_RETAINED_DECISIONS.to_string(),
        "--minibatch",
        "512",
    ])
    .expect("settings");
    let mut environments: Vec<_> = (0..2)
        .map(|side| fixture_environment(TICK_CAP - 24, side, OpponentSpec::Weak))
        .collect();
    let mut states = [EpisodeStream::default(), EpisodeStream::default()];
    let mut random = actor_stream_rngs(&mut PpoRng::new(982_003), 2).expect("streams");
    let mut rollout =
        PpoRollout::new(2, model.policy_identity().expect("identity")).expect("rollout");
    let mut report = PpoSmokeReport::default();
    for _ in 0..8 {
        let (choices, spaces) =
            sample_active(&model, &mut random, &mut environments, &[0, 1]).expect("sample");
        for (stream, (choice, space)) in choices.into_iter().zip(spaces).enumerate() {
            advance_stream(
                &model,
                &mut environments[stream],
                &mut states[stream],
                (stream, choice, space),
                settings.ppo,
                &mut rollout,
                &mut report,
            )
            .expect("advance");
        }
    }
    assert!(
        states
            .iter()
            .all(|state| state.done && state.choice.is_none())
    );
    assert_eq!(report.terminal_draws, 2);
    assert_eq!(report.terminal_losses, 0);
    assert_eq!(report.terminal_wins, 0);
    assert_eq!(report.episode_timeouts, 0);
    assert_eq!(report.elapsed_ticks, 48);
    validate_episode_batch(&rollout, &report).expect("all-draw batch is usable");
    assert_eq!(rollout.len(), 2);
    let batch = rollout.finish(settings.ppo).expect("draw batch");
    for index in 0..2 {
        let sample = batch.sample(index).expect("one sample per seat");
        assert_eq!(sample.transition.stream, index);
        assert_eq!(sample.transition.ticks, 24);
        assert!(sample.transition.terminal);
        assert_eq!(sample.transition.next_value, 0.0);
        assert!(sample.return_value().is_finite());
    }
}

#[test]
fn map2_final_damage_is_rewarded_before_draw_finish_in_window_collection() {
    let model = stop_model();
    let mut environment = fixture_environment(TICK_CAP - 1, 0, OpponentSpec::Weak);
    environment.arena.configure_for_test(|world| {
        let source = world.seats[0].unit.expect("source");
        let target = world.seats[1].unit.expect("target");
        world.push_hit(Some(source), target, 100, DamageKind::Pure);
    });
    let config = PpoConfig {
        environments: 1,
        rollout_decisions: 1,
        minibatch: 1,
        gamma_tick: 1.0,
        ..PpoConfig::default()
    };
    let mut environments = vec![environment];
    let mut pending = collect_round(
        &model,
        &mut [PpoRng::new(982_004)],
        &mut environments,
        config,
    )
    .expect("final round");
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].ticks, 1);
    assert_eq!(pending[0].terminal_outcome, Some(PpoTerminalOutcome::Draw));
    assert!(pending[0].reward > 0.0);
    assert!(pending[0].terminal);
    assert!(pending.remove(0).next_frame.is_none());
    assert_eq!(
        environments[0].seats[0]
            .tracker
            .finish_map2_reward(Map2RewardEnd::Draw)
            .expect_err("already finalized")
            .to_string(),
        "Map2 reward: episode already ended"
    );
}

#[test]
fn map2_reward_interval_and_full_episode_aggregates_preserve_every_component() {
    let mut split = fixture_environment(1, 0, OpponentSpec::Weak);
    let mut whole = fixture_environment(1, 0, OpponentSpec::Weak);
    let mut aggregate = super::super::Map2TrainingReward::default();
    for tick in 2..=9 {
        for environment in [&mut split, &mut whole] {
            let seat = &mut environment.seats[0];
            let mut view = seat.tracker.current().expect("view").clone();
            view.tick = tick;
            let own = seat.tracker.own_hero().expect("own").id;
            let enemy = view
                .players
                .iter()
                .find(|row| row.team != seat.tracker.team())
                .expect("enemy row")
                .unit
                .expect("enemy");
            let events = vec![damage(own, enemy, 10), damage(enemy, own, 5)];
            observe_messages(
                seat,
                &[
                    ServerMsg::Snapshot { view },
                    ServerMsg::Events { tick, events },
                ],
            )
            .expect("full tick");
        }
        if tick < 9 {
            aggregate
                .record(
                    split.seats[0]
                        .tracker
                        .take_map2_reward_interval()
                        .expect("interval"),
                )
                .expect("aggregate");
        }
    }
    aggregate
        .record(
            split.seats[0]
                .tracker
                .finish_map2_reward(Map2RewardEnd::Draw)
                .expect("final interval"),
        )
        .expect("aggregate final");
    let mut expected = super::super::Map2TrainingReward::default();
    expected
        .record(
            whole.seats[0]
                .tracker
                .finish_map2_reward(Map2RewardEnd::Draw)
                .expect("full episode"),
        )
        .expect("whole aggregate");
    assert_eq!(aggregate.ticks, 8);
    assert_eq!(aggregate.observations, expected.observations);
    for (left, right) in aggregate
        .components()
        .into_iter()
        .zip(expected.components())
    {
        assert!((left - right).abs() < 1e-12);
    }
    assert_eq!(
        split.seats[0].tracker.map2_reward_state(),
        whole.seats[0].tracker.map2_reward_state()
    );
}

#[test]
fn map2_complete_collection_accepts_only_draws_and_flushes_random_phase_partial_intervals() {
    let model = stop_model();
    let settings = crate::cli::training_settings_for_test(&[
        "--environments",
        "2",
        "--rollout",
        &crate::MAP2_RETAINED_DECISIONS.to_string(),
        "--minibatch",
        "512",
    ])
    .expect("settings");
    let phases: Vec<_> = streams_for_collection(&settings, 0)
        .expect("phases")
        .iter()
        .map(|state| state.retention_phase)
        .collect();
    assert!(phases.iter().any(|phase| *phase > 0));
    let mut environments: Vec<_> = (0..2)
        .map(|side| fixture_environment(TICK_CAP - 24, side, OpponentSpec::Weak))
        .collect();
    let mut rollout =
        PpoRollout::new(2, model.policy_identity().expect("identity")).expect("rollout");
    let mut report = PpoSmokeReport::default();
    collect(
        &model,
        &mut PpoRng::new(982_009),
        &mut environments,
        &settings,
        0,
        &mut rollout,
        &mut report,
    )
    .expect("all native draws");
    assert_eq!(rollout.len(), 2);
    assert_eq!(report.terminal_draws, 2);
    assert_eq!(report.map2_reward.ticks, 48);
    assert!(report.map2_reward.total < 0.0);
    assert_eq!(report.episode_timeouts, 0);
    let batch = rollout
        .finish(settings.ppo)
        .expect("retained partial intervals");
    for (stream, phase) in phases.into_iter().enumerate() {
        let sample = batch.sample(stream).expect("sample");
        assert_eq!(sample.transition.ticks, (8 - phase) as u32 * 3);
        assert!(sample.transition.terminal);
        assert_eq!(sample.transition.next_value, 0.0);
    }
}

#[test]
fn map2_learner_time_cap_finishes_reward_and_zero_bootstraps_without_inventing_match_over() {
    let model = stop_model();
    let mut environment = fixture_environment(1, 0, OpponentSpec::Weak);
    let choice =
        sample_policy(&model, &mut PpoRng::new(982_010), &mut environment).expect("action");
    let mut state = EpisodeStream {
        choice: Some(choice.clone()),
        done: true,
        decisions: 1,
        ..EpisodeStream::default()
    };
    let advanced = advance_interval(&mut environment, vec![None, None], 3).expect("live ticks");
    assert_eq!(advanced.winner, None);
    let reward =
        observe_reward(&mut environment, &mut state, None, 3, 1.0).expect("learner deadline");
    state
        .append_retained_reward(reward, 3, 1.0)
        .expect("pending interval");
    let mut rollout = PpoRollout::new(1, choice.policy()).expect("rollout");
    let mut report = PpoSmokeReport::default();
    finish_advance(
        &model,
        &mut environment,
        &mut state,
        0,
        CompletedAdvance {
            end_tick: 4,
            ticks: 3,
            outcome: None,
        },
        &mut rollout,
        &mut report,
    )
    .expect("terminal deadline");
    validate_episode_batch(&rollout, &report).expect("task deadline is valid optimizer data");
    assert_eq!(report.episode_timeouts, 1);
    assert_eq!(report.terminal_draws, 0);
    assert_eq!(state.map2_reward.terminal, 0.0);
    assert_eq!(rollout.len(), 1);
    let config = PpoConfig {
        environments: 1,
        rollout_decisions: 1,
        minibatch: 1,
        gamma_tick: 1.0,
        ..PpoConfig::default()
    };
    let sample = rollout
        .finish(config)
        .expect("batch")
        .sample(0)
        .expect("sample");
    assert!(sample.transition.terminal);
    assert_eq!(sample.transition.next_value, 0.0);
    assert_eq!(sample.transition.ticks, 3);
}

#[test]
fn map2_training_reward_diagnostics_distinguish_cost_components_from_raw_units() {
    let mut report = super::super::Map2TrainingReward::default();
    report
        .record(crate::Map2RewardBreakdown {
            ticks: 3,
            hero_damage: 0.01,
            mana_spent: -0.002,
            hero_damage_taken: -0.001,
            observations: crate::Map2RewardObservations {
                hero_damage_dealt: 67,
                hero_damage_taken: 39,
                mana_spent: 75,
                lane_last_hits: 2,
                ..crate::Map2RewardObservations::default()
            },
            ..crate::Map2RewardBreakdown::default()
        })
        .expect("report");
    let output = report.to_string();
    for field in [
        "reward_hero_damage=0.010000000",
        "reward_mana=-0.002000000",
        "reward_hero_taken=-0.001000000",
        "hero_damage_dealt=67",
        "hero_damage_taken=39",
        "mana_spent=75",
        "lane_last_hits=2",
    ] {
        assert!(
            output.split_whitespace().any(|entry| entry == field),
            "{field}: {output}"
        );
    }
    assert!(output.len() < 4096);
}

#[test]
fn map2_training_reward_aggregation_rejects_overflow_and_nonfinite_values_atomically() {
    for component in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut report = super::super::Map2TrainingReward::default();
        assert_eq!(
            report
                .record(crate::Map2RewardBreakdown {
                    total: component,
                    ..crate::Map2RewardBreakdown::default()
                })
                .expect_err("invalid component")
                .to_string(),
            "PPO Map2 reward telemetry is non-finite"
        );
        assert_eq!(report, super::super::Map2TrainingReward::default());
    }
    let mut report = super::super::Map2TrainingReward {
        observations: crate::Map2RewardObservations {
            mana_spent: u64::MAX,
            ..crate::Map2RewardObservations::default()
        },
        ..super::super::Map2TrainingReward::default()
    };
    let before = report;
    let error = report
        .record(crate::Map2RewardBreakdown {
            observations: crate::Map2RewardObservations {
                mana_spent: 1,
                ..crate::Map2RewardObservations::default()
            },
            ..crate::Map2RewardBreakdown::default()
        })
        .expect_err("counter overflow");
    assert_eq!(error, PpoError::CounterOverflow);
    assert_eq!(report, before);
}

#[test]
fn map2_neural_candidate_and_opponent_can_use_mango_then_cast_without_server_rejection() {
    use crate::{ActionTarget, ControlledUnit, StructuredAction};
    use bota_proto::{AbilitySlot, Fixed, ItemId, ItemSlot};
    let mut environment = configured_environment(1, 0, OpponentSpec::Weak, |world| {
        for index in 0..2 {
            let seat = &world.seats[index];
            let hero = seat.unit.expect("hero");
            world.inventory.get_mut(hero).expect("inventory").slots[0] =
                bota_server::game::ItemStack::bought(ItemId(42), seat.slot, 1);
            world.mana.get_mut(hero).expect("mana").mana = Fixed::ZERO;
            world.abilities.get_mut(hero).expect("abilities").slots[0].level = 1;
        }
    });
    let actions = [
        StructuredAction::Use {
            unit: ControlledUnit::Hero,
            slot: ItemSlot(0),
            target: ActionTarget::None,
        },
        StructuredAction::Cast {
            unit: ControlledUnit::Hero,
            slot: AbilitySlot(0),
            target: ActionTarget::None,
        },
    ];
    for action in actions {
        let mut requests = Vec::with_capacity(2);
        for seat in &mut environment.seats {
            let (_, space) = prepare_neural_seat_policy_sample(seat).expect("pure Neural space");
            assert!(space.allows(action));
            requests.push(
                neural_policy_request_in_space(seat, action, &space)
                    .expect("raw Neural transport")
                    .1,
            );
        }
        let advanced = advance_interval(&mut environment, requests, 3).expect("native response");
        assert_eq!(advanced.winner, None);
        reject_production_rejection(&environment, "Map2 Mango and raze contract")
            .expect("no NotReady");
        let reward = take_map2_reward(&mut environment, None, 3).expect("full event streams");
        if action.kind() == ActionKind::Use {
            assert_eq!(reward.observations.mana_spent, 0);
            for seat in &environment.seats {
                let hero = seat.tracker.own_hero().expect("restored hero");
                assert!(hero.mana >= 100);
                assert!(hero.items[0].is_none());
            }
        } else {
            assert!(reward.observations.mana_spent > 0);
            assert!(reward.mana_spent < 0.0);
        }
    }
    assert_eq!(
        environment_rejections(&[environment]).expect("both seats"),
        0
    );
}

#[test]
fn map2_received_damage_and_attacker_credit_can_differ_when_only_the_victim_receives_the_event() {
    let mut environment = configured_environment(1, 0, OpponentSpec::Weak, |world| {
        let source = world.seats[0].unit.expect("source");
        let target = world.seats[1].unit.expect("target");
        world
            .transform
            .get_mut(source)
            .expect("source position")
            .pos = Vec2::from_ints(2_500, 2_500);
        world
            .transform
            .get_mut(target)
            .expect("target position")
            .pos = Vec2::from_ints(12_000, 9_000);
        world.push_hit(Some(source), target, 39, DamageKind::Pure);
    });
    let advanced =
        advance_interval(&mut environment, vec![None, None], 1).expect("one native tick");
    assert_eq!(advanced.winner, None);
    let attacker = environment.seats[0]
        .tracker
        .take_map2_reward_interval()
        .expect("attacker-visible reward");
    let receiver = environment.seats[1]
        .tracker
        .take_map2_reward_interval()
        .expect("receiver-visible reward");
    assert_eq!(attacker.observations.hero_damage_dealt, 0);
    assert_eq!(receiver.observations.hero_damage_taken, 39);
    assert_eq!(attacker.ticks, receiver.ticks);
}

#[test]
fn map2_warmup_drains_discarded_credit_without_replenishing_lifetime_reward_budgets() {
    let mut environment = configured_environment(1, 0, OpponentSpec::Weak, |world| {
        world.push_hit(
            world.seats[1].unit,
            world.seats[0].unit.expect("hero"),
            100,
            DamageKind::Pure,
        );
    });
    advance_interval(&mut environment, vec![None, None], 1).expect("warmup damage");
    let before = environment.seats[0]
        .tracker
        .map2_reward_state()
        .expect("reward");
    assert!(before.remaining[5] < 1.0);
    finish_warmup_environment(&mut environment, 3).expect("discard warmup rewards");
    let after = environment.seats[0]
        .tracker
        .map2_reward_state()
        .expect("preserved reward");
    assert_eq!(after.remaining, before.remaining);
    assert_eq!(after.completed_tick, Some(8));
    for seat in &mut environment.seats {
        assert_eq!(
            seat.tracker
                .take_map2_reward_interval()
                .expect("drained credit"),
            crate::Map2RewardBreakdown::default()
        );
    }
}

fn damage(source: EntityId, target: EntityId, amount: i32) -> EventKind {
    EventKind::Damaged {
        source: Some(source),
        target,
        amount,
        kind: DamageKind::Pure,
        crit: false,
    }
}

fn fixture_environment(
    tick: u32,
    policy_seat: usize,
    opponent_spec: OpponentSpec,
) -> TrainingEnvironment {
    configured_environment(tick, policy_seat, opponent_spec, |_| {})
}

fn configured_environment(
    tick: u32,
    policy_seat: usize,
    opponent_spec: OpponentSpec,
    configure: impl FnOnce(&mut bota_server::game::World),
) -> TrainingEnvironment {
    assert!(tick > 0);
    assert!(tick < TICK_CAP);
    let (mut arena, mut start) = Arena::new(ArenaConfig {
        seats: 2,
        map: MapId(2),
        seed: 982_001,
    })
    .expect("Map2 arena");
    let configured = arena.configure_for_test(|world| {
        world.tick = tick;
        for (side, seat) in world.seats.iter().enumerate() {
            let unit = seat.unit.expect("hero");
            world.transform.get_mut(unit).expect("position").pos =
                Vec2::from_ints(9_216 + side as i32 * 80, 9_216);
        }
        configure(world);
    });
    for (messages, current) in start.messages.iter_mut().zip(configured.messages) {
        messages.truncate(1);
        messages.extend(current);
    }
    TrainingEnvironment {
        arena,
        seats: setup_seats(start).expect("full baseline"),
        policy_seat,
        reward: RewardTracker::default(),
        decision: 0,
        map: MapId(2),
        next_seed: 982_005,
        next_opponent_seed: 982_006,
        opponent: build_opponent(&opponent_spec, 982_007).expect("opponent"),
        opponent_spec,
        retired_rejections: 0,
    }
}

fn stop_model() -> PolicyModel {
    let model = PolicyModel::fresh(982_008).expect("model");
    let mut parameters = vec![0.0; crate::MODEL_PARAMETER_COUNT];
    let mut offset = 0;
    for (name, shape) in model.parameter_schema().expect("schema") {
        if name == "kind.bias" {
            parameters[offset + ActionKind::Stop.index()] = 100.0;
        }
        offset += shape.iter().product::<usize>();
    }
    assert_eq!(offset, parameters.len());
    model.import_parameters(&parameters).expect("Stop policy");
    model
}
