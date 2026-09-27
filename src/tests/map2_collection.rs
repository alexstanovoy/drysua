//! Native collection and concrete reward/accounting regressions, not helper layouts.

use super::*;
use crate::{Map2RewardEnd, PpoBatch};
use bota_proto::{DamageKind, Vec2};

#[test]
fn reward_rejection_preserves_accumulated_credit_and_allows_recovery() {
    let interval = crate::Map2RewardBreakdown {
        ticks: 3,
        gold: 0.02,
        total: 0.02,
        ..crate::Map2RewardBreakdown::default()
    };
    let mut aggregate = Map2TrainingReward::default();
    aggregate.record(interval).expect("initial credit");
    let before = aggregate;
    for total in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let error = aggregate
            .record(crate::Map2RewardBreakdown { total, ..interval })
            .expect_err("nonfinite reward");
        assert_eq!(error.to_string(), "PPO Map2 reward telemetry is non-finite");
        assert_eq!(aggregate, before);
    }
    aggregate.record(interval).expect("recovery");
    assert_eq!(aggregate.ticks, 6);
    assert_eq!(aggregate.gold, 0.04);
    aggregate.observations.stagnation_repaid_ticks = u64::MAX;
    let before = aggregate;
    let mut overflow = interval;
    overflow.observations.stagnation_repaid_ticks = 1;
    assert_eq!(aggregate.record(overflow), Err(PpoError::CounterOverflow));
    assert_eq!(aggregate, before);
}

#[test]
fn native_batches_preserve_curriculum_mastery_and_terminal_retention() {
    for (count, groups, schedule, update) in [
        (6, 1, "weak-warmup-v1", 0),
        (6, 1, "weak-warmup-v1", 2),
        (26, 1, "mastery-v1", 0),
        (26, 4, "mastery-v1", 0),
    ] {
        let count_text = count.to_string();
        let groups_text = groups.to_string();
        let rollout_text = crate::MAP2_RETAINED_DECISIONS.to_string();
        let mut arguments = vec![
            "--opponent-schedule",
            schedule,
            "--environments",
            &count_text,
            "--pipeline-groups",
            &groups_text,
            "--rollout",
            &rollout_text,
            "--minibatch",
            "512",
        ];
        if schedule == "mastery-v1" {
            arguments.extend(["--mastery-window", "3"]);
        }
        let settings =
            crate::cli::training_settings_for_test(&arguments).expect("complete episode settings");
        let model = stop_model();
        let batch = native_batch(&model, &settings, update);
        assert_eq!(batch.len(), count);
        let phases = streams_for_collection(&settings, update).expect("retention phases");
        let mut negative = false;
        for index in 0..count {
            let sample = batch.sample(index).expect("sample");
            assert!(sample.transition.terminal);
            assert_eq!(sample.transition.next_value, 0.0);
            assert_eq!(
                sample.transition.ticks,
                (8 - phases[sample.transition.stream].retention_phase) as u32 * 3
            );
            assert!(sample.return_value().is_finite());
            assert!(sample.return_value() <= 0.0);
            negative |= sample.return_value() < 0.0;
        }
        assert!(negative, "native all-draw batch retains real dense cost");
    }
}

fn native_batch(model: &PolicyModel, settings: &TrainingJobConfig, update: u64) -> PpoBatch {
    let count = settings.ppo.environments;
    let mut arenas: Vec<_> = (0..count)
        .map(|stream| {
            let opponent = if settings.mastery_config.is_some() {
                OpponentSpec::Weak
            } else {
                scheduled_opponent(settings, update, stream)
            };
            configured_environment(TICK_CAP - 24, stream % 2, opponent, |_| {})
        })
        .collect();
    let opponents = if update == 2 {
        ("Mixed", 4, 2)
    } else {
        ("Weak", count, 0)
    };
    assert_eq!(collection_opponents(&arenas), opponents);
    let mut random = PpoRng::new(9140300);
    let mut rollout = PpoRollout::new(
        count * RETAINED_PER_EPISODE,
        model.policy_identity().expect("identity"),
    )
    .expect("rollout");
    let mut report = PpoSmokeReport::default();
    collect_groups_bounded(
        model,
        &mut random,
        &mut arenas,
        settings,
        update,
        settings.pipeline_groups,
        ACTOR_DECISIONS,
        true,
        None,
        &mut rollout,
        &mut report,
    )
    .expect("native terminal batch");
    assert_eq!(report.terminal_draws, count as u64);
    assert_eq!(report.terminal_losses, 0);
    assert_eq!(report.terminal_wins, 0);
    assert_eq!(report.episode_timeouts, 0);
    assert_eq!(report.rejected_orders, 0);
    assert_eq!(report.elapsed_ticks, (count * 24) as u64);
    assert_eq!(report.map2_reward.ticks, report.elapsed_ticks);
    assert_eq!(report.map2_reward.terminal, 0.0);
    if let Some(config) = settings.mastery_config {
        let mut mastery = crate::MasteryProgress::default();
        assert!(
            !mastery
                .record_batch(config, &report.completed_episodes.ordered_outcomes())
                .expect("outcomes")
        );
        assert_eq!(mastery.games(), count as u64);
        assert_eq!(mastery.wins(), 0);
        assert_eq!(mastery.stage(), crate::MasteryStage::Weak);
    }
    rollout.finish(settings.ppo).expect("usable PPO batch")
}

#[test]
fn progress_debt_and_purchase_refund_survive_reward_drains() {
    let environment = configured_environment(1, 0, OpponentSpec::Weak, |_| {});
    let mut tracker = environment.seats[0].tracker.clone();
    let mut view = tracker.current().expect("snapshot").clone();
    let own = tracker.own_hero().expect("hero").id;
    let hero = view
        .units
        .iter_mut()
        .find(|unit| unit.id == own)
        .expect("hero");
    hero.hp = hero.max_hp;
    hero.mana = hero.max_mana;
    hero.effects
        .retain(|effect| effect.id != bota_proto::EffectId(3));
    let position = hero.pos;
    view.units
        .iter_mut()
        .find(|unit| unit.kind == bota_proto::UnitKind::Fountain && unit.team == tracker.team())
        .expect("fountain")
        .pos = position;
    let mut aggregate = Map2TrainingReward::default();
    for tick in 2..=2705 {
        view.tick = tick;
        if tick == 2704 {
            view.players[0].xp += 1;
        }
        let events = if tick == 2703 {
            vec![EventKind::ItemBought {
                slot: tracker.slot(),
                item: bota_proto::ItemId(0),
            }]
        } else {
            vec![]
        };
        tracker.observe_snapshot(&view).expect("snapshot");
        tracker.observe_events(tick, &events).expect("events");
        if tick % 17 == 0 {
            aggregate
                .record(tracker.take_map2_reward_interval().expect("drain"))
                .expect("aggregate");
        }
    }
    aggregate
        .record(
            tracker
                .finish_map2_reward(Map2RewardEnd::Draw)
                .expect("split end"),
        )
        .expect("final");
    assert_eq!(aggregate.ticks, 2704);
    let components = aggregate.components();
    assert!(
        (components[..components.len() - 1].iter().sum::<f64>() - aggregate.total).abs() < 1e-12
    );
    assert_eq!(
        aggregate.observations.progress_reasons,
        crate::MAP2_PROGRESS_PURCHASE | crate::MAP2_PROGRESS_XP
    );
    assert_eq!(aggregate.observations.stagnation_base_charges, 1);
    assert_eq!(aggregate.stagnation_base, -0.02);
    assert_eq!(aggregate.stagnation_ticks_cost, -0.000002);
    assert!(aggregate.fountain_wait_refund > 0.0);
    assert_eq!(aggregate.observations.fountain_wait_refunds, 1);
}

#[test]
fn both_seats_consume_untruncated_events_once_and_reject_reordered_terminal_messages() {
    let mut environment = configured_environment(TICK_CAP - 2, 0, OpponentSpec::Weak, |_| {});
    let heroes: Vec<_> = environment.seats[0]
        .tracker
        .current()
        .expect("view")
        .players
        .iter()
        .map(|row| row.unit.expect("hero"))
        .collect();
    for (index, seat) in environment.seats.iter_mut().enumerate() {
        let mut view = seat.tracker.current().expect("view").clone();
        view.tick += 1;
        let tick = view.tick;
        let events = vec![
            EventKind::Damaged {
                source: Some(heroes[index]),
                target: heroes[1 - index],
                amount: 1,
                kind: DamageKind::Pure,
                crit: false,
            };
            300
        ];
        observe_messages(
            seat,
            &[
                ServerMsg::Snapshot { view },
                ServerMsg::Events { tick, events },
            ],
        )
        .expect("full event batch");
        let reward = seat.tracker.take_map2_reward_interval().expect("reward");
        assert_eq!(reward.observations.hero_damage_dealt, 300);
        assert_eq!(reward.ticks, 1);
        assert_eq!(
            seat.tracker
                .take_map2_reward_interval()
                .expect("drained")
                .ticks,
            0
        );
    }
    let mut environment = configured_environment(TICK_CAP - 1, 0, OpponentSpec::Weak, |_| {});
    let mut messages = environment
        .arena
        .step(&[None, None])
        .expect("final tick")
        .messages
        .remove(0);
    messages.swap(1, 2);
    let seat = &mut environment.seats[0];
    let before = seat.tracker.map2_reward_state();
    assert_eq!(
        observe_messages(seat, &messages)
            .expect_err("reordered terminal")
            .to_string(),
        "invalid PPO transition: arena Snapshot/Events/MatchOver ordering"
    );
    assert_eq!(seat.tracker.map2_reward_state(), before);
}

#[test]
fn final_native_damage_is_rewarded_before_draw_finalization() {
    let model = stop_model();
    let mut environment = configured_environment(TICK_CAP - 1, 0, OpponentSpec::Weak, |_| {});
    environment.arena.configure_for_test(|world| {
        world.push_hit(
            world.seats[0].unit,
            world.seats[1].unit.expect("target"),
            100,
            DamageKind::Pure,
        );
    });
    let config = PpoConfig {
        environments: 1,
        rollout_decisions: 1,
        minibatch: 1,
        gamma_tick: 1.0,
        ..PpoConfig::default()
    };
    let mut arenas = vec![environment];
    let mut pending = collect_round(&model, &mut [PpoRng::new(982_004)], &mut arenas, config)
        .expect("final round");
    assert_eq!(pending.len(), 1);
    let final_row = pending.remove(0);
    assert_eq!(final_row.ticks, 1);
    assert_eq!(final_row.terminal_outcome, Some(PpoTerminalOutcome::Draw));
    let reward = final_row.map2_reward.expect("Map2 reward");
    assert_eq!(reward.terminal, 0.0);
    assert!(reward.hero_damage > 0.0);
    assert!(final_row.reward > 0.0);
    assert!(final_row.terminal);
    assert!(final_row.next_frame.is_none());
    assert_eq!(
        arenas[0].seats[0]
            .tracker
            .finish_map2_reward(Map2RewardEnd::Draw)
            .expect_err("already finalized")
            .to_string(),
        "Map2 reward: episode already ended"
    );
}

#[test]
fn learner_deadline_zero_bootstraps_without_inventing_match_over() {
    let model = stop_model();
    let mut environment = configured_environment(1, 0, OpponentSpec::Weak, |_| {});
    let choice =
        sample_policy(&model, &mut PpoRng::new(982_010), &mut environment).expect("action");
    let mut state = EpisodeStream {
        choice: Some(choice.clone()),
        done: true,
        decisions: 1,
        ..EpisodeStream::default()
    };
    assert_eq!(
        advance_interval(&mut environment, vec![None, None], 3)
            .expect("live ticks")
            .winner,
        None
    );
    let reward = observe_reward(&mut environment, &mut state, None, 3, 1.0).expect("deadline");
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
    validate_episode_batch(&rollout, &report).expect("usable deadline batch");
    assert_eq!(report.episode_timeouts, 1);
    assert_eq!(report.terminal_draws, 0);
    assert_eq!(state.map2_reward.terminal, -0.2);
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
fn native_mango_then_cast_is_legal_for_both_neural_seats() {
    use crate::{ActionTarget, ControlledUnit, StructuredAction};
    use bota_proto::{AbilitySlot, Fixed, ItemId, ItemSlot};
    let mut environment = configured_environment(1, 0, OpponentSpec::Weak, |world| {
        for seat in &world.seats {
            let hero = seat.unit.expect("hero");
            world.inventory.get_mut(hero).expect("inventory").slots[0] =
                bota_server::game::ItemStack::bought(ItemId(42), seat.slot, 1);
            world.mana.get_mut(hero).expect("mana").mana = Fixed::ZERO;
            world.abilities.get_mut(hero).expect("abilities").slots[0].level = 1;
        }
    });
    for action in [
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
    ] {
        let mut requests = Vec::with_capacity(2);
        for seat in &mut environment.seats {
            let (_, space) = prepare_neural_seat_policy_sample(seat).expect("Neural space");
            assert!(space.allows(action));
            requests.push(
                neural_policy_request_in_space(seat, action, &space)
                    .expect("transport")
                    .1,
            );
        }
        assert_eq!(
            advance_interval(&mut environment, requests, 3)
                .expect("native response")
                .winner,
            None
        );
        reject_production_rejection(&environment, "Map2 Mango and raze contract")
            .expect("no NotReady");
        let reward = take_map2_reward(&mut environment, None, 3).expect("event streams");
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
fn victim_only_damage_and_warmup_drains_preserve_seat_credit_and_lifetime_budgets() {
    let mut environment = configured_environment(1, 1, OpponentSpec::Weak, |world| {
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
    assert_eq!(
        advance_interval(&mut environment, vec![None, None], 1)
            .expect("native tick")
            .winner,
        None
    );
    let attacker = environment.seats[0]
        .tracker
        .take_map2_reward_interval()
        .expect("attacker reward");
    let victim = environment.seats[1]
        .tracker
        .take_map2_reward_interval()
        .expect("victim reward");
    assert_eq!(attacker.observations.hero_damage_dealt, 0);
    assert_eq!(victim.observations.hero_damage_taken, 39);
    assert_eq!(attacker.ticks, victim.ticks);
    let before = environment.seats[1]
        .tracker
        .map2_reward_state()
        .expect("reward");
    assert!(before.remaining[5] < 1.0);
    finish_warmup_environment(&mut environment, 3).expect("discard warmup rewards");
    let after = environment.seats[1]
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
            world
                .transform
                .get_mut(seat.unit.expect("hero"))
                .expect("position")
                .pos = Vec2::from_ints(9_216 + side as i32 * 80, 9_216);
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
