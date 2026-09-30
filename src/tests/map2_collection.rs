//! Native collection and concrete reward/accounting regressions, not helper layouts.

use super::*;
use crate::Map2RewardEnd;
use bota_proto::{DamageKind, Vec2};

#[test]
fn reward_rejection_preserves_accumulated_credit_and_allows_recovery() {
    let interval = crate::Map2RewardBreakdown {
        ticks: 3,
        health: 0.02,
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
    assert_eq!(aggregate.components[2], 0.04);
    aggregate.observations.hero_damage_dealt = u64::MAX;
    let before = aggregate;
    let mut overflow = interval;
    overflow.observations.hero_damage_dealt = 1;
    assert_eq!(aggregate.record(overflow), Err(PpoError::CounterOverflow));
    assert_eq!(aggregate, before);
}

#[test]
fn both_seats_consume_untruncated_events_once_and_reject_reordered_terminal_messages() {
    let mut environment = configured_environment(TICK_CAP - 2, 0, OpponentRuntime::Idle, |_| {});
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
    let mut environment = configured_environment(TICK_CAP - 1, 0, OpponentRuntime::Idle, |_| {});
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

/// One policy decision against an idle opponent, booked like a slot does.
fn step_policy(
    environment: &mut TrainingEnvironment,
    stream: &mut EpisodeStream,
    model: &PolicyModel,
    done: bool,
) -> CompletedAdvance {
    let (frame, space) = prepare_policy_sample(environment).expect("frame");
    let choice = model
        .sample(&frame, &space, &mut PpoRng::new(982_004))
        .expect("action");
    if stream.retains(choice.action().kind()) {
        stream.retain(RetainedChoice {
            frame,
            target: choice.target.clone(),
            action: choice.action(),
            behaviour: 0,
            log_probability: choice.log_probability(),
            value: choice.value(),
        });
    }
    let seat = environment.policy_seat;
    let mut requests = vec![None, None];
    requests[seat] =
        neural_policy_request_in_space(&mut environment.seats[seat], choice.action(), &space)
            .expect("policy request")
            .1;
    requests[1 - seat] =
        scripted_request(&mut environment.seats[1 - seat], OpponentRuntime::Idle).expect("idle");
    let tick = environment.arena.tick();
    let stepped = advance_interval(environment, requests, 3.min(TICK_CAP - tick)).expect("ticks");
    let outcome = terminal_outcome(environment, stepped.winner);
    let end_tick = tick + stepped.ticks;
    let advanced = CompletedAdvance {
        end_tick,
        ticks: stepped.ticks,
        outcome,
        done: done || outcome.is_some() || end_tick >= TICK_CAP,
    };
    stream
        .record_decision(environment, choice.action().kind(), &advanced, 1.0)
        .expect("booked decision");
    advanced
}

#[test]
fn final_native_damage_is_rewarded_before_draw_finalization() {
    let model = stop_model();
    let mut environment = configured_environment(TICK_CAP - 1, 0, OpponentRuntime::Idle, |_| {});
    environment.arena.configure_for_test(|world| {
        world.push_hit(
            world.seats[0].unit,
            world.seats[1].unit.expect("target"),
            100,
            DamageKind::Pure,
        );
    });
    let mut stream = EpisodeStream::new();
    let completed = step_policy(&mut environment, &mut stream, &model, false);
    assert_eq!(completed.ticks, 1);
    assert_eq!(completed.outcome, Some(PpoTerminalOutcome::Draw));
    assert!(stream.done());
    let reward = stream.map2_reward();
    assert_eq!(reward.components[5], 0.0);
    assert!(reward.components[2] > 0.0);
    assert_eq!(reward.observations.hero_damage_dealt, 100);
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
    let mut environment = configured_environment(1, 0, OpponentRuntime::Idle, |_| {});
    let mut stream = EpisodeStream::new();
    let completed = step_policy(&mut environment, &mut stream, &model, true);
    assert_eq!(completed.outcome, None);
    assert_eq!((completed.end_tick, completed.ticks), (4, 3));
    let transition = stream.flush(None).expect("terminal deadline");
    assert!(transition.terminal);
    assert_eq!(transition.next_value, 0.0);
    assert_eq!(transition.ticks, 3);
    assert_eq!(stream.map2_reward().components[5], 0.0);
    let summary = crate::ppo_arena::game_summary::GameSummary::capture(&environment, None, 4);
    let record = stream.record(
        0,
        0,
        &completed,
        crate::ppo_arena::slot::OpponentKind::Teacher,
        &summary,
    );
    let mut report = CollectionReport::default();
    record.accumulate(&mut report).expect("report");
    assert_eq!(report.episode_timeouts, 1);
    assert_eq!(report.terminal_draws, 0);
}

#[test]
fn native_mango_then_aimed_raze_is_legal_for_both_neural_seats() {
    use crate::{ActionTarget, ControlledUnit, StructuredAction};
    use bota_proto::{AbilitySlot, Fixed, ItemId, ItemSlot};
    let mut environment = configured_environment(1, 0, OpponentRuntime::Idle, |world| {
        for (index, seat) in world.seats.iter().enumerate() {
            let hero = seat.unit.expect("hero");
            world.inventory.get_mut(hero).expect("inventory").slots[0] =
                bota_server::game::ItemStack::bought(ItemId(42), seat.slot, 1);
            world.mana.get_mut(hero).expect("mana").mana = Fixed::ZERO;
            world.abilities.get_mut(hero).expect("abilities").slots[0].level = 1;
            // Each hero faces the other, so the aimed near raze goes off at once.
            world.transform.get_mut(hero).expect("facing").facing.brads =
                if index == 0 { 0 } else { 32_768 };
        }
    });
    for kind in [ActionKind::Use, ActionKind::Cast] {
        let mut requests = Vec::with_capacity(2);
        for seat in &mut environment.seats {
            let (_, space) = prepare_neural_seat_policy_sample(seat).expect("Neural space");
            let action = if kind == ActionKind::Use {
                StructuredAction::Use {
                    unit: ControlledUnit::Hero,
                    slot: ItemSlot(0),
                    target: ActionTarget::None,
                }
            } else {
                let enemy = seat.tracker.current().expect("view").players
                    [usize::from(seat.tracker.team() == bota_proto::Team::Radiant)]
                .unit
                .expect("enemy hero");
                StructuredAction::Cast {
                    unit: ControlledUnit::Hero,
                    slot: AbilitySlot(0),
                    target: ActionTarget::Entity(space.entity_index(enemy).expect("enemy")),
                }
            };
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
        take_map2_reward(&mut environment, None, 3).expect("event streams");
        for seat in &environment.seats {
            let hero = seat.tracker.own_hero().expect("hero");
            if kind == ActionKind::Use {
                assert!(hero.mana >= 100);
                assert!(hero.items[0].is_none());
            } else {
                assert!(hero.mana < hero.max_mana);
            }
        }
    }
    assert!(environment.seats.iter().all(|seat| seat.rejections == 0));
}

#[test]
fn victim_only_damage_preserves_seat_credit() {
    let mut environment = configured_environment(1, 1, OpponentRuntime::Idle, |world| {
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
    opponent_spec: OpponentRuntime,
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
        opponent: opponent_spec,
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
