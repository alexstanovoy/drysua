use super::*;

#[cfg(test)]
pub(crate) struct PpoOrderContractProbe(ArenaSeatPolicy);

#[cfg(test)]
impl PpoOrderContractProbe {
    pub(crate) fn new(side: usize, messages: &[ServerMsg]) -> Self {
        Self(setup_seat(side, messages).expect("PPO order-contract seat"))
    }

    pub(crate) fn observe(&mut self, messages: &[ServerMsg]) {
        assert!(
            observe_messages(&mut self.0, messages)
                .expect("PPO probe observations")
                .is_none()
        );
    }

    pub(crate) fn decide(
        &mut self,
        model: &PolicyModel,
        candidate: bool,
    ) -> (FeatureFrame, Option<Request>, Option<ActivePolicyOrder>) {
        let (frame, space) = if candidate {
            prepare_neural_seat_policy_sample(&mut self.0)
        } else {
            prepare_seat_policy_sample(&mut self.0)
        }
        .expect("PPO probe input");
        let action = model
            .choose(&frame, &space)
            .expect("PPO probe choice")
            .action;
        let request = if candidate {
            // Only the learner's request transport is tested; these placeholder
            // statistics never enter a rollout, reward calculation or optimizer.
            let choice = PpoPolicyChoice {
                frame: frame.clone(),
                action,
                target: crate::BehavioralTarget::from_action(&frame, &space, action)
                    .expect("probe target"),
                policy: model.policy_identity().expect("probe policy"),
                log_probability: 0.0,
                entropy: 0.0,
                value: 0.0,
            };
            policy_request_in_space(&mut self.0, &choice, &space).expect("PPO learner transport")
        } else {
            neural_policy_request_in_space(&mut self.0, action, &space)
                .expect("legacy probe request")
                .1
        };
        (frame, request, self.0.local.active_order())
    }

    pub(crate) fn legacy_persistence(&self) -> OrderPersistence {
        self.0.persistence
    }
}

pub(crate) fn merge_actor_report_for_test(
    aggregate: &mut PpoSmokeReport,
    actor: PpoSmokeReport,
) -> Result<(), PpoError> {
    merge_actor_report(aggregate, actor)
}

#[cfg(test)]
pub(crate) const fn training_warmup_phase_index(stream: u64) -> usize {
    (training_pair_index(stream) % 8) as usize
}

#[cfg(test)]
pub(crate) fn assert_frozen_neural_opponent_for_test(model: &PolicyModel) {
    let snapshot = PolicySnapshot::capture(model, 0).expect("frozen snapshot");
    for map in [MapId(0), MapId(1), MapId(2)] {
        let mut environment = build_environment(
            23_090,
            23_091,
            map,
            0,
            0,
            OpponentSpec::Policy(snapshot.clone()),
        )
        .expect("frozen opponent arena");
        assert!(matches!(
            environment.opponent,
            OpponentRuntime::Policy { .. }
        ));
        for _ in 0..4 {
            let (requests, kind) = requests_for_neural_greedy_decision(&mut environment, model)
                .expect("pure requests");
            assert_eq!(kind, ActionKind::Stop);
            assert!(requests.iter().flatten().all(|request| matches!(
                request.order,
                bota_proto::Order::Move {
                    target: bota_proto::Target::None
                }
            )));
            advance_interval(&mut environment, requests, 3).expect("pure ticks");
        }
        assert_eq!(environment.seats[0].sequence, 1);
        assert!((1..=4).contains(&environment.seats[1].sequence));
    }
}

#[cfg(test)]
pub(crate) fn assert_pure_warmup_ledger_for_test(model: &PolicyModel, kind: ActionKind) {
    for map in [MapId(0), MapId(1), MapId(2)] {
        for side in 0..2 {
            let mut batched = vec![
                build_environment(23_088, 23_089, map, side, 0, OpponentSpec::Teacher)
                    .expect("arena"),
            ];
            let mut raw = build_environment(23_088, 23_089, map, side, 0, OpponentSpec::Teacher)
                .expect("arena");
            for _ in 0..4 {
                let requests = requests_for_batched_greedy_decisions(&mut batched, &[0], model)
                    .expect("warmup");
                let (expected, action) =
                    requests_for_neural_greedy_decision(&mut raw, model).expect("raw neural");
                assert_eq!(action, kind);
                assert_eq!(requests, vec![expected.clone()]);
                advance_warmup_environments(&mut batched, &[0], requests, 3).expect("warmup ticks");
                advance_interval(&mut raw, expected, 3).expect("raw ticks");
                for seat in 0..2 {
                    assert_eq!(batched[0].seats[seat].sequence, raw.seats[seat].sequence);
                    assert_eq!(
                        batched[0].seats[seat].tracker.latest_summary(),
                        raw.seats[seat].tracker.latest_summary()
                    );
                    assert_eq!(batched[0].seats[seat].rejections, 0);
                }
            }
            assert!(raw.seats[1 - side].sequence > 0);
            assert!(raw.seats[side].sequence <= 1);
        }
    }
}

#[cfg(test)]
fn run_warmup_decisions(
    environment: &mut TrainingEnvironment,
    model: &PolicyModel,
    decisions: usize,
    decision_interval_ticks: u32,
) -> Result<(), PpoError> {
    for _ in 0..decisions {
        let (requests, _) = requests_for_neural_greedy_decision(environment, model)?;
        let advanced = advance_interval(environment, requests, decision_interval_ticks)?;
        reject_production_rejection(environment, "production warmup")?;
        if advanced.winner.is_some() {
            restart_environment(environment)?;
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn production_warmup_order_counts_for_test(
    model: &PolicyModel,
    decisions: usize,
) -> Result<[u32; 2], PpoError> {
    let mut environment = build_environment(23_078, 23_079, MapId(1), 0, 0, OpponentSpec::Weak)?;
    run_warmup_decisions(&mut environment, model, decisions, 3)?;
    assert_eq!(environment.policy_seat, 0);
    Ok([environment.seats[0].sequence, environment.seats[1].sequence])
}

#[cfg(test)]
pub(crate) fn production_batched_warmup_order_counts_for_test(
    model: &PolicyModel,
    decisions: usize,
) -> Result<[[u32; 2]; 2], PpoError> {
    let mut environments = vec![
        build_environment(23_082, 23_083, MapId(1), 0, 0, OpponentSpec::Weak)?,
        build_environment(23_082, 23_083, MapId(1), 1, 0, OpponentSpec::Weak)?,
    ];
    for _ in 0..decisions {
        let active = [0, 1];
        let requests = requests_for_batched_greedy_decisions(&mut environments, &active, model)?;
        advance_warmup_environments(&mut environments, &active, requests, 3)?;
    }
    Ok(std::array::from_fn(|environment| {
        std::array::from_fn(|seat| environments[environment].seats[seat].sequence)
    }))
}

#[cfg(test)]
pub(crate) fn production_warmup_cleanup_preserves_readiness_for_test() -> Result<bool, PpoError> {
    let mut environment = build_environment(23_080, 23_081, MapId(1), 0, 0, OpponentSpec::Weak)?;
    for seat in &mut environment.seats {
        seat.readiness
            .note_shared_wait_for_test(crate::ControlledUnit::Hero, 7, 2_100);
    }
    let before = environment
        .seats
        .iter()
        .map(|seat| seat.readiness)
        .collect::<Vec<_>>();
    clear_warmup_orders(&mut environment, 3)?;
    Ok(environment
        .seats
        .iter()
        .zip(before)
        .all(|(seat, readiness)| seat.readiness == readiness))
}

#[cfg(test)]
pub(crate) fn rejection_delta_for_test(before: u64, after: u64) -> Result<u64, PpoError> {
    rejection_delta(before, after)
}

#[cfg(test)]
pub(crate) const fn deployment_uses_teacher_for_test(map: MapId) -> bool {
    deployment_uses_teacher(map)
}

#[cfg(test)]
pub(super) fn observe_messages(
    seat: &mut ArenaSeatPolicy,
    messages: &[ServerMsg],
) -> Result<Option<Team>, PpoError> {
    observe_messages_owned(seat, messages.to_vec())
}
