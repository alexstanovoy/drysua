//! Native collection and concrete reward/accounting regressions, not helper layouts.

use super::*;
use crate::Map2RewardEnd;
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
fn progress_debt_and_purchase_refund_survive_reward_drains() {
    let environment = configured_environment(1, 0, OpponentSpec::Idle, |_| {});
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
    let mut environment = configured_environment(TICK_CAP - 2, 0, OpponentSpec::Idle, |_| {});
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
    let mut environment = configured_environment(TICK_CAP - 1, 0, OpponentSpec::Idle, |_| {});
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
    let mut environment = configured_environment(TICK_CAP - 1, 0, OpponentSpec::Idle, |_| {});
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
    let (frame, space) = prepare_policy_sample(&mut environment).expect("final frame");
    let choice = model
        .sample(&frame, &space, &mut PpoRng::new(982_004))
        .expect("final action");
    let mut state = EpisodeStream::default();
    let completed =
        advance_cpu(&mut environment, &mut state, choice, space, config).expect("final round");
    assert_eq!(completed.ticks, 1);
    assert_eq!(completed.outcome, Some(PpoTerminalOutcome::Draw));
    assert!(state.done);
    assert_eq!(state.map2_reward.terminal, 0.0);
    assert!(state.map2_reward.hero_damage > 0.0);
    assert!(state.raw_return > 0.0);
    assert_eq!(
        environment.seats[0]
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
    let mut environment = configured_environment(1, 0, OpponentSpec::Idle, |_| {});
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
    let mut report = CollectionReport::default();
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
    let mut environment = configured_environment(1, 0, OpponentSpec::Idle, |world| {
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
fn victim_only_damage_preserves_seat_credit() {
    let mut environment = configured_environment(1, 1, OpponentSpec::Idle, |world| {
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
}

pub(super) fn configured_environment(
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
        map: MapId(2),
        opponent: build_opponent(&opponent_spec, 982_007).expect("opponent"),
    }
}

fn stop_model() -> PolicyModel {
    let model = PolicyModel::fresh(982_008).expect("model");
    let mut parameters = vec![0.0; crate::MODEL_PARAMETER_COUNT];
    let mut offset = 0;
    for (name, shape) in model.parameter_schema().expect("schema") {
        if matches!(name, "kind.bias" | "dire.kind.bias") {
            parameters[offset + ActionKind::Stop.index()] = 100.0;
        }
        offset += shape.iter().product::<usize>();
    }
    assert_eq!(offset, parameters.len());
    model.import_parameters(&parameters).expect("Stop policy");
    model
}
