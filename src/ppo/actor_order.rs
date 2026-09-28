use super::*;

const _: () = assert!(PPO_WIDE_ANNEALED_MAX_SAMPLES <= 93_040);
const _: () = assert!(PPO_WIDE_ANNEALED_MAX_GAMES <= 80);

impl PpoRollout {
    /// Restore sequential actor-batch order before GAE/normalization, without moving
    /// ragged rows. Stable buckets preserve the original order within each batch.
    pub(crate) fn canonicalize_actor_order(
        &mut self,
        config: PpoConfig,
        batch_width: usize,
    ) -> Result<(), PpoError> {
        let config = config.validate()?;
        if self.sample_budget != config.sample_budget {
            return Err(PpoError::InvalidConfig("rollout sample budget"));
        }
        if config.environments > PPO_WIDE_ANNEALED_MAX_GAMES
            || !(1..=PPO_ANNEALED_MAX_PARALLEL_WORLDS).contains(&batch_width)
            || !config.environments.is_multiple_of(batch_width)
        {
            return Err(PpoError::InvalidConfig("actor rollout ordering dimensions"));
        }
        let maximum = config
            .environments
            .checked_mul(config.rollout_decisions)
            .ok_or(PpoError::InvalidConfig("actor rollout ordering samples"))?;
        if self.len() > maximum
            || self.len() > self.capacity
            || self.len() > PPO_WIDE_ANNEALED_MAX_SAMPLES
        {
            return Err(PpoError::InvalidConfig("actor rollout ordering samples"));
        }
        // Both descriptor validation and the only allocation finish before any record moves.
        let mut destinations =
            actor_destinations(&self.transitions, config.environments, batch_width)?;
        apply_actor_order(&mut self.transitions, &mut destinations);
        Ok(())
    }
}

fn actor_destinations(
    transitions: &[CompactPpoTransition],
    environments: usize,
    batch_width: usize,
) -> Result<Vec<usize>, PpoError> {
    assert!((1..=PPO_WIDE_ANNEALED_MAX_GAMES).contains(&environments));
    assert!(batch_width > 0);
    let mut next = [0usize; PPO_WIDE_ANNEALED_MAX_GAMES];
    for transition in transitions {
        if transition.stream >= environments {
            return Err(PpoError::StreamOutOfRange {
                stream: transition.stream,
            });
        }
        next[transition.stream / batch_width] += 1;
    }
    let mut total = 0;
    for count in &mut next {
        let start = total;
        total += *count;
        *count = start;
    }
    assert_eq!(total, transitions.len());
    let mut destinations = actor_order_indices(total)?;
    for transition in transitions {
        let destination = &mut next[transition.stream / batch_width];
        destinations.push(*destination);
        *destination += 1;
    }
    Ok(destinations)
}

fn actor_order_indices(count: usize) -> Result<Vec<usize>, PpoError> {
    if count > PPO_WIDE_ANNEALED_MAX_SAMPLES {
        return Err(PpoError::InvalidConfig("actor rollout ordering samples"));
    }
    #[cfg(test)]
    if FAIL_ALLOCATION.get() {
        return Err(PpoError::InvalidTransition(
            "actor rollout ordering allocation failed",
        ));
    }
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(count)
        .map_err(|_| PpoError::InvalidTransition("actor rollout ordering allocation failed"))?;
    if indices.capacity() > PPO_WIDE_ANNEALED_MAX_SAMPLES {
        return Err(PpoError::InvalidTransition(
            "actor rollout ordering allocation exceeds capacity",
        ));
    }
    Ok(indices)
}

fn apply_actor_order(transitions: &mut [CompactPpoTransition], destinations: &mut [usize]) {
    assert_eq!(transitions.len(), destinations.len());
    assert!(transitions.len() <= PPO_WIDE_ANNEALED_MAX_SAMPLES);
    let mut swaps_left = transitions.len();
    for index in 0..transitions.len() {
        // Every swap fixes its destination; a permutation needs fewer than N swaps in total.
        let maximum_swaps = swaps_left;
        for _ in 0..maximum_swaps {
            let destination = destinations[index];
            if destination == index {
                break;
            }
            assert!(destination > index);
            assert!(destination < transitions.len());
            transitions.swap(index, destination);
            destinations.swap(index, destination);
            swaps_left -= 1;
        }
        assert_eq!(destinations[index], index);
    }
}

#[cfg(test)]
thread_local! { static FAIL_ALLOCATION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actor_order_preserves_stable_bucket_order_frames_returns_and_push_continuity() {
        let sample = sample();
        let config = config(8);
        let initial = [(3, 0), (0, 0), (2, 0), (1, 0)];
        let later = [
            (7, 0),
            (4, 0),
            (6, 0),
            (5, 0),
            (3, 1),
            (0, 1),
            (2, 1),
            (1, 1),
        ];
        let mut actual = rollout(&sample, config, &initial);
        let frames = frames(&actual);
        let counters = actual.next_decision;

        actual
            .canonicalize_actor_order(config, 2)
            .expect("first wave");

        assert_eq!(order(&actual), [(0, 0), (1, 0), (3, 0), (2, 0)]);
        assert_eq!(actual.next_decision, counters);
        for (stream, decision, frame) in frames {
            let row = actual
                .transitions
                .iter()
                .find(|row| (row.stream, row.decision) == (stream, decision))
                .expect("row");
            assert_eq!(actual.frames.expand(&row.frame).expect("frame"), frame);
        }
        let first = order(&actual);
        actual
            .canonicalize_actor_order(config, 2)
            .expect("idempotent");
        assert_eq!(order(&actual), first);
        for &(stream, decision) in &later {
            actual
                .push(row(&sample, stream, decision))
                .expect("continued push");
        }
        actual
            .canonicalize_actor_order(config, 2)
            .expect("later global buckets");
        let expected = [
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (3, 0),
            (2, 0),
            (3, 1),
            (2, 1),
            (4, 0),
            (5, 0),
            (7, 0),
            (6, 0),
        ];
        assert_eq!(order(&actual), expected);
        let reference = rollout(&sample, config, &expected)
            .finish(config)
            .expect("reference");
        let actual = actual.finish(config).expect("canonical GAE");
        crate::ppo_arena::episode::assert_batch_parity_for_test(&reference, &actual);
    }

    #[test]
    fn actor_order_handles_empty_buckets_last_stream_and_scratch_boundaries() {
        let sample = sample();
        let config = config(80);
        let mut actual = rollout(&sample, config, &[(79, 0), (0, 0), (79, 1)]);
        actual
            .canonicalize_actor_order(config, 1)
            .expect("eighty buckets");
        assert_eq!(order(&actual), [(0, 0), (79, 0), (79, 1)]);
        let mut empty = rollout(&sample, config, &[]);
        empty
            .canonicalize_actor_order(config, 40)
            .expect("empty order");
        assert!(empty.is_empty());
        for count in [0, 1, PPO_WIDE_ANNEALED_MAX_SAMPLES] {
            let scratch = actor_order_indices(count).expect("bounded scratch");
            assert!(scratch.is_empty());
            assert!(scratch.capacity() >= count);
            assert!(scratch.capacity() <= PPO_WIDE_ANNEALED_MAX_SAMPLES);
        }
        for count in [PPO_WIDE_ANNEALED_MAX_SAMPLES + 1, usize::MAX] {
            assert_eq!(
                actor_order_indices(count),
                Err(PpoError::InvalidConfig("actor rollout ordering samples"))
            );
        }
    }

    #[test]
    fn actor_order_invalid_dimensions_stream_and_allocation_leave_rows_unchanged() {
        let sample = sample();
        let config = config(4);
        let mut actual = rollout(&sample, config, &[(3, 0), (0, 0), (2, 0), (1, 0)]);
        let before = order(&actual);
        for width in [0, 3, 65, usize::MAX] {
            let error = actual
                .canonicalize_actor_order(config, width)
                .expect_err("invalid width");
            assert_eq!(
                error.to_string(),
                "invalid PPO config field: actor rollout ordering dimensions"
            );
            assert_eq!(order(&actual), before);
        }
        let wrong_budget = PpoConfig {
            sample_budget: PpoSampleBudget::Annealed,
            ..config
        };
        assert_eq!(
            actual.canonicalize_actor_order(wrong_budget, 2),
            Err(PpoError::InvalidConfig("rollout sample budget"))
        );
        assert_eq!(order(&actual), before);
        actual.transitions.last_mut().expect("last row").stream = 4;
        let corrupt = order(&actual);
        assert_eq!(
            actual.canonicalize_actor_order(config, 2),
            Err(PpoError::StreamOutOfRange { stream: 4 })
        );
        assert_eq!(order(&actual), corrupt);
        actual.transitions.last_mut().expect("last row").stream = 1;
        FAIL_ALLOCATION.set(true);
        let result = actual.canonicalize_actor_order(config, 2);
        FAIL_ALLOCATION.set(false);
        assert_eq!(
            result,
            Err(PpoError::InvalidTransition(
                "actor rollout ordering allocation failed"
            ))
        );
        assert_eq!(order(&actual), before);
    }

    fn config(games: usize) -> PpoConfig {
        PpoConfig {
            environments: games,
            rollout_decisions: crate::MAP2_RETAINED_DECISIONS,
            sample_budget: PpoSampleBudget::for_annealed_games(games),
            gamma_tick: 1.0,
            ..PpoConfig::default()
        }
    }

    fn rollout(sample: &PpoTransition, config: PpoConfig, rows: &[(usize, u32)]) -> PpoRollout {
        let mut rollout =
            PpoRollout::with_budget(16, sample.policy, config.sample_budget).expect("rollout");
        for &(stream, decision) in rows {
            rollout.push(row(sample, stream, decision)).expect("push");
        }
        rollout
    }

    fn row(sample: &PpoTransition, stream: usize, decision: u32) -> PpoTransition {
        let mut row = sample.clone();
        row.stream = stream;
        row.decision = decision;
        row.reward = stream as f32 / 16.0 + decision as f32;
        row.frame.global[0] = row.reward;
        row.frame.units[3][crate::unit_feature::TOKEN_PRESENT] = 1.0;
        row.frame.units[3][1] = row.reward;
        row.terminal = decision == 1;
        row.next_value = if row.terminal { 0.0 } else { 0.25 };
        row
    }

    fn order(rollout: &PpoRollout) -> Vec<(usize, u32)> {
        rollout
            .transitions
            .iter()
            .map(|row| (row.stream, row.decision))
            .collect()
    }

    fn frames(rollout: &PpoRollout) -> Vec<(usize, u32, FeatureFrame)> {
        rollout
            .transitions
            .iter()
            .map(|row| {
                (
                    row.stream,
                    row.decision,
                    rollout.frames.expand(&row.frame).expect("frame"),
                )
            })
            .collect()
    }

    fn sample() -> PpoTransition {
        use bota_proto::{MapId, ServerMsg, SlotId};
        let (_, start) = crate::Arena::new(crate::ArenaConfig {
            seats: 2,
            map: MapId(0),
            seed: 9952700,
        })
        .expect("arena");
        let [
            ServerMsg::MatchStart { info },
            ServerMsg::Snapshot { view },
            ServerMsg::Events { tick, events },
        ] = start.messages[0].as_slice()
        else {
            panic!("native start");
        };
        let mut tracker = crate::StateTracker::new(SlotId(0), info).expect("tracker");
        tracker.observe_snapshot(view).expect("snapshot");
        tracker.observe_events(*tick, events).expect("events");
        let space = crate::ActionSpace::from_tracker(&tracker).expect("space");
        let mut encoder = crate::FeatureEncoder::new(&tracker);
        encoder.observe(&tracker).expect("observe");
        let mut frame = FeatureFrame::new();
        encoder
            .encode(
                &tracker,
                &space,
                &crate::ItemReadiness::new(),
                &crate::LocalPolicyState::new(0),
                &mut frame,
            )
            .expect("encode");
        let model = PolicyModel::fresh(9952700).expect("model");
        let target = BehavioralTarget::from_action(&frame, &space, StructuredAction::Continue)
            .expect("target");
        PpoTransition {
            frame,
            target,
            action: StructuredAction::Continue,
            policy: model.policy_identity().expect("policy"),
            stream: 0,
            decision: 0,
            ticks: 3,
            old_log_probability: -0.5,
            old_value: 0.1,
            next_value: 0.25,
            reward: 1.0,
            terminal: false,
        }
    }
}
