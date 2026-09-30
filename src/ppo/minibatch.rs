pub(super) fn partition(
    order: &[usize],
    maximum: usize,
    balanced: bool,
) -> impl Iterator<Item = &[usize]> {
    assert!(
        order.len() <= super::PPO_MAX_SAMPLES,
        "PPO partition row count {} exceeds maximum {}",
        order.len(),
        super::PPO_MAX_SAMPLES
    );
    assert!((1..=super::MODEL_MAX_BATCH).contains(&maximum));
    let chunks = order.chunks(maximum);
    let count = chunks.len();
    chunks.enumerate().map(move |(index, chunk)| {
        if !balanced {
            return chunk;
        }
        let size = order.len() / count;
        let larger = order.len() % count;
        let start = index * size + index.min(larger);
        let end = start + size + usize::from(index < larger);
        &order[start..end]
    })
}

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn partition_full_capacity_preserves_every_row_and_order_in_both_modes() {
        let order: Vec<_> = (0..PPO_MAX_SAMPLES).rev().collect();
        assert_eq!(order.len(), 33_280);
        for balanced in [false, true] {
            let chunks = minibatch::partition(&order, 2048, balanced).collect::<Vec<_>>();
            assert_eq!(chunks.len(), order.len().div_ceil(2048));
            assert_eq!(
                chunks.iter().map(|chunk| chunk.len()).sum::<usize>(),
                order.len()
            );
            assert!(
                chunks
                    .iter()
                    .all(|chunk| !chunk.is_empty() && chunk.len() <= 2048)
            );
            assert_eq!(chunks.concat(), order);
            if balanced {
                let smallest = chunks.iter().map(|chunk| chunk.len()).min().unwrap();
                let largest = chunks.iter().map(|chunk| chunk.len()).max().unwrap();
                assert!(largest - smallest <= 1);
            } else {
                assert!(chunks.into_iter().eq(order.chunks(2048)));
            }
        }
    }

    #[test]
    fn partition_full_capacity_rejects_maximum_plus_one_with_the_bound() {
        let order = vec![0; PPO_MAX_SAMPLES + 1];
        assert_eq!(order.len(), 33_281);
        for balanced in [false, true] {
            let panic =
                std::panic::catch_unwind(|| minibatch::partition(&order, 2048, balanced).count())
                    .expect_err("partition must reject above the largest admitted PPO capacity");
            let message = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied());
            assert_eq!(
                message,
                Some("PPO partition row count 33281 exceeds maximum 33280")
            );
        }
    }

    #[test]
    fn balanced_partitions_preserve_order_and_bound_sizes_without_a_tiny_tail() {
        let order = [8, 2, 6, 0, 7, 1, 5, 3, 4, 9];
        for (count, sizes) in [
            (0, &[][..]),
            (1, &[1][..]),
            (4, &[4][..]),
            (5, &[3, 2][..]),
            (8, &[4, 4][..]),
            (9, &[3, 3, 3][..]),
            (10, &[4, 3, 3][..]),
        ] {
            let slices = minibatch::partition(&order[..count], 4, true).collect::<Vec<_>>();
            let lengths = slices.iter().map(|slice| slice.len()).collect::<Vec<_>>();
            assert_eq!(lengths, sizes);
            assert_eq!(slices.concat(), order[..count]);
            assert!(minibatch::partition(&order[..count], 4, false).eq(order[..count].chunks(4)));
        }
    }

    #[test]
    fn short_default_and_balanced_updates_cover_every_row_with_different_boundaries() {
        let config = PpoConfig {
            samples_per_update: 9,
            minibatch: 4,
            epochs: 2,
            target_kl: 1000.0,
            ..PpoConfig::default()
        };
        let mut results = Vec::new();
        for balanced in [false, true] {
            let model = PolicyModel::fresh(9101).unwrap();
            let mut trainer = PpoTrainer::new(&model, config, 19).unwrap();
            trainer
                .set_execution(crate::TrainingExecutionOptions {
                    balanced_minibatches: balanced,
                    ..Default::default()
                })
                .unwrap();
            let batch = sampled_batch(&model, config);
            let before = model.export_parameters().unwrap();
            let frame = batch.sample(0).unwrap().transition.frame;
            let value_before = model.evaluate(&frame).unwrap().value;
            let report = trainer
                .train_update(&model, &batch, crate::UpdateObjective::default())
                .unwrap();
            let after = model.export_parameters().unwrap();
            assert_eq!(report.samples_optimized, 18);
            assert_eq!(report.minibatches, 6);
            assert_eq!(report.optimizer_step, 6);
            assert_ne!(before, after);
            let order: Vec<_> = (0..9).collect();
            let sizes: Vec<_> = minibatch::partition(&order, 4, balanced)
                .map(<[_]>::len)
                .collect();
            eprintln!(
                "minibatch-comparison source=synthetic_fixed_state_terminal_rewards_1_to_9 seed=9101 balanced={balanced} sizes={sizes:?} epochs=2 rows_optimized={} adam_steps={} value_before={value_before} value_after={} policy_loss={} value_loss={} gradient_norm={} applied_scale={} shuffle={:?}",
                report.samples_optimized,
                report.optimizer_step,
                model.evaluate(&frame).unwrap().value,
                report.policy_loss,
                report.value_loss,
                report.gradient_norm,
                report.applied_scale,
                trainer.rng_checkpoint()
            );
            results.push((before, after, trainer.rng_checkpoint()));
        }
        assert_eq!(results[0].0, results[1].0);
        assert_eq!(results[0].2, results[1].2);
        assert_ne!(results[0].1, results[1].1);
        let difference = results[0]
            .1
            .iter()
            .zip(&results[1].1)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        eprintln!(
            "minibatch-comparison max_parameter_difference={difference} identical_work=true numerical_equivalence=false"
        );
    }

    #[test]
    fn balanced_trainer_rolls_back_a_failed_update_and_replays_the_restored_batch() {
        let config = PpoConfig {
            samples_per_update: 9,
            minibatch: 4,
            epochs: 2,
            target_kl: 1000.0,
            ..PpoConfig::default()
        };
        let model = PolicyModel::fresh(9101).expect("model");
        let mut trainer = PpoTrainer::new(&model, config, 19).expect("trainer");
        trainer
            .set_execution(crate::TrainingExecutionOptions {
                balanced_minibatches: true,
                ..crate::TrainingExecutionOptions::default()
            })
            .expect("balanced execution");
        let mut batch = sampled_batch(&model, config);
        let before = trainer.checkpoint_snapshot(&model).expect("snapshot");
        let mut shuffle = PpoRng::new(19);
        let mut order = (0..9).collect::<Vec<_>>();
        shuffle.shuffle(&mut order).expect("first epoch order");
        let tail = order[8];
        let frame = batch.samples[tail].transition.frame.clone();
        batch.corrupt_materialization_frame_for_test(tail);
        let error = trainer
            .train_update(&model, &batch, crate::UpdateObjective::default())
            .expect_err("tail error");
        assert_eq!(
            error.to_string(),
            "invalid PPO transition: ragged feature range is invalid"
        );
        let restored = trainer.checkpoint_snapshot(&model).expect("rollback");
        assert_eq!(restored.parameters, before.parameters);
        assert_eq!(restored.adam.moments(), before.adam.moments());
        assert_eq!(restored.adam.config(), before.adam.config());
        assert_eq!(restored.adam.step(), before.adam.step());
        assert_eq!(trainer.rng_checkpoint(), PpoRng::new(19).checkpoint());
        assert_eq!(trainer.updates(), 0);
        // A restored batch replays the rolled-back update from identical state.
        batch.samples[tail].transition.frame = frame;
        let report = trainer
            .train_update(&model, &batch, crate::UpdateObjective::default())
            .expect("restored rollout");
        assert_eq!(report.samples_optimized, 18);
        assert_eq!(report.minibatches, 6);
        assert_eq!((report.optimizer_step, trainer.optimizer_step()), (6, 6));
        assert_eq!(report.epochs_completed, 2);
        assert_eq!(trainer.updates(), 1);
        shuffle.shuffle(&mut order).expect("second epoch order");
        assert_eq!(trainer.rng_checkpoint(), shuffle.checkpoint());
        let after = trainer.checkpoint_snapshot(&model).expect("learned");
        assert_ne!(after.parameters, before.parameters);
    }

    fn sampled_batch(model: &PolicyModel, config: PpoConfig) -> PpoBatch {
        let choice = choice(model);
        let mut rollout = PpoRollout::new(config.samples_per_update).expect("rollout");
        for decision in 0..9 {
            let transition = choice
                .clone()
                .finish(
                    0,
                    PpoOutcome {
                        stream: 0,
                        decision,
                        ticks: 3,
                        next_value: 0.0,
                        reward: decision as f32 + 1.0,
                        terminal: true,
                    },
                )
                .expect("transition");
            rollout.push(transition).expect("push");
        }
        rollout.finish(config).expect("batch")
    }

    fn choice(model: &PolicyModel) -> PpoPolicyChoice {
        use bota_proto::{MapId, MatchInfo, Pick, PlayerView, SlotId, Team, TickMode, WorldView};
        let info = MatchInfo {
            match_id: 1,
            map: MapId(0),
            tick_rate: 30,
            pregame_ticks: 0,
            trees: Vec::new(),
            terrain_cells: 1,
            terrain_rle: vec![(1, 0x80)],
            opaque_cells: Vec::new(),
            mode: TickMode::Lockstep,
            picks: vec![Pick {
                slot: SlotId(0),
                team: Team::Radiant,
                hero: crate::SHADOW_FIEND,
            }],
            shop: Vec::new(),
            fountains: [bota_proto::Vec2::ZERO; 2],
            shop_range: 0,
        };
        let mut tracker = crate::StateTracker::new(SlotId(0), &info).expect("tracker");
        let view = WorldView {
            tick: 1,
            viewer: Some(Team::Radiant),
            units: Vec::new(),
            projectiles: Vec::new(),
            players: vec![PlayerView {
                slot: SlotId(0),
                team: Team::Radiant,
                hero: crate::SHADOW_FIEND,
                unit: None,
                level: 1,
                xp: 0,
                gold: Some(0),
                stash: Some(vec![None; 6]),
                kit: None,
                kills: 0,
                deaths: 0,
                assists: 0,
                last_hits: 0,
                denies: 0,
                respawn_left: 1,
            }],
            felled_trees: Vec::new(),
            planted_trees: Vec::new(),
            loot: Vec::new(),
        };
        tracker.observe_snapshot(&view).expect("snapshot");
        let space = crate::ActionSpace::from_tracker(&tracker).expect("action space");
        let mut encoder = crate::FeatureEncoder::new(&tracker);
        encoder.observe(&tracker).expect("observation");
        let mut frame = FeatureFrame::new();
        encoder
            .encode(
                &tracker,
                &space,
                &crate::ItemReadiness::new(),
                &crate::LocalPolicyState::new(0),
                &mut frame,
            )
            .expect("frame");
        model
            .sample(&frame, &space, &mut PpoRng::new(5))
            .expect("choice")
    }
}
