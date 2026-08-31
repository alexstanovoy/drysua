use std::sync::{Arc, Barrier};
use std::thread;

use bota_proto::Team;

use super::feature::{encode, tracker_with_view, world_view};
use crate::{
    ActorLearnerPipeline, ActorPolicyLease, ActorRolloutBuffer, LocalPolicyState, PPO_MAX_SAMPLES,
    PipelineError, PolicyDevice, PolicyModel, PpoConfig, PpoOutcome, PpoRng, PpoTrainer,
    RolloutVersion,
};

#[test]
fn explicit_cpu_model_matches_default_initialization() {
    let default_model = PolicyModel::fresh(17_001).expect("default model");
    let cpu_model = PolicyModel::fresh_on(17_001, PolicyDevice::Cpu).expect("CPU model");

    assert_eq!(cpu_model.device(), PolicyDevice::Cpu);
    assert_eq!(
        default_model
            .export_parameters()
            .expect("default parameters"),
        cpu_model.export_parameters().expect("CPU parameters")
    );
}

#[test]
fn fixed_worker_endpoints_and_sample_capacity_reject_boundaries() {
    let learner = PolicyModel::fresh(17_010).expect("learner");

    assert!(matches!(
        ActorLearnerPipeline::new(0, 1, &learner),
        Err(PipelineError::InvalidSampleCapacity { capacity: 0 })
    ));
    assert!(matches!(
        ActorLearnerPipeline::new(PPO_MAX_SAMPLES + 1, 1, &learner),
        Err(PipelineError::InvalidSampleCapacity { .. })
    ));
    assert!(matches!(
        ActorLearnerPipeline::new(1, 0, &learner),
        Err(PipelineError::InvalidWorkerCount { workers: 0 })
    ));

    let mut pipeline = ActorLearnerPipeline::new(1, 1, &learner).expect("pipeline");
    let actor = pipeline.take_actor(0).expect("worker zero");
    let lease = actor.lease().expect("actor policy");
    let Err(error) = PpoTrainer::new(lease.policy(), single_sample_config(), 17_011) else {
        panic!("actor model claimed an optimizer");
    };
    assert_eq!(
        error.to_string(),
        "PPO model error: immutable actor model cannot own an optimizer"
    );
    assert!(matches!(
        pipeline.take_actor(0),
        Err(PipelineError::WorkerAlreadyTaken { worker: 0 })
    ));
    assert!(matches!(
        pipeline.take_actor(1),
        Err(PipelineError::WorkerOutOfRange {
            worker: 1,
            workers: 1
        })
    ));
}

#[test]
fn two_buffer_ownership_accepts_one_generation_lag_and_rejects_third_rollout() {
    let learner = PolicyModel::fresh(17_020).expect("learner");
    let mut pipeline = ActorLearnerPipeline::new(1, 2, &learner).expect("pipeline");
    let actor_a = pipeline.take_actor(0).expect("actor A");
    let actor_b = pipeline.take_actor(1).expect("actor B");
    let lease_a = actor_a.lease().expect("lease A");
    let lease_b = actor_b.lease().expect("lease B");
    let rollout_a = sample_rollout(&lease_a, 1, 1);
    let rollout_b = sample_rollout(&lease_b, 2, 1);
    lease_a.try_submit(rollout_a).expect("ready buffer A");
    lease_b.try_submit(rollout_b).expect("ready buffer B");
    let third_lease = actor_a.lease().expect("third lease metadata");
    assert!(matches!(
        third_lease.rollout(),
        Err(PipelineError::BuffersExhausted)
    ));
    let accepted_a = pipeline.try_accept().expect("learner owns A");

    advance_parameters(&learner);
    assert_eq!(
        pipeline.publish(&learner).expect("publish"),
        RolloutVersion::new(1)
    );
    let accepted_b = pipeline.try_accept().expect("one-generation lag");

    assert_eq!(accepted_a.rollout_version(), RolloutVersion::new(0));
    assert_eq!(accepted_b.rollout_version(), RolloutVersion::new(0));
    assert_eq!(accepted_b.learner_version(), RolloutVersion::new(1));
}

#[test]
fn actor_fills_second_buffer_while_learner_holds_first_buffer() {
    let learner = PolicyModel::fresh(17_025).expect("learner");
    let mut pipeline = ActorLearnerPipeline::new(1, 2, &learner).expect("pipeline");
    let actor_a = pipeline.take_actor(0).expect("actor A");
    let actor_b = pipeline.take_actor(1).expect("actor B");
    let lease_a = actor_a.lease().expect("lease A");
    let rollout_a = sample_rollout(&lease_a, 10, 1);
    lease_a.try_submit(rollout_a).expect("submit A");
    let learner_buffer = pipeline.try_accept().expect("learner holds A");
    let start = Arc::new(Barrier::new(2));
    let complete = Arc::new(Barrier::new(2));
    let actor_start = Arc::clone(&start);
    let actor_complete = Arc::clone(&complete);
    let worker = thread::spawn(move || {
        let lease_b = actor_b.lease().expect("lease B");
        let rollout_b = sample_rollout(&lease_b, 11, 1);
        actor_start.wait();
        lease_b.try_submit(rollout_b).expect("submit B");
        actor_complete.wait();
    });

    start.wait();
    complete.wait();
    let ready_buffer = pipeline.try_accept().expect("actor progressed");

    assert_eq!(learner_buffer.rollout_version(), RolloutVersion::new(0));
    assert_eq!(ready_buffer.rollout_version(), RolloutVersion::new(0));
    worker.join().expect("actor worker");
}

#[test]
fn actor_collects_second_rollout_during_actual_learner_update() {
    let learner = PolicyModel::fresh(17_027).expect("learner");
    let mut pipeline = ActorLearnerPipeline::new(1, 2, &learner).expect("pipeline");
    let actor_a = pipeline.take_actor(0).expect("actor A");
    let actor_b = pipeline.take_actor(1).expect("actor B");
    let lease_a = actor_a.lease().expect("lease A");
    let rollout_a = sample_rollout(&lease_a, 12, 1);
    lease_a.try_submit(rollout_a).expect("submit A");
    let batch_a = pipeline
        .try_accept()
        .expect("accept A")
        .finish(single_sample_config())
        .expect("batch A");
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let learner_entered = Arc::clone(&entered);
    let learner_release = Arc::clone(&release);
    let mut trainer = PpoTrainer::new(&learner, single_sample_config(), 17_028).expect("trainer");

    thread::scope(|scope| {
        let update = scope.spawn(|| {
            trainer
                .train_pipeline_update_with_barriers_for_test(
                    &learner,
                    &batch_a,
                    &learner_entered,
                    &learner_release,
                )
                .expect("learner update")
        });
        entered.wait();
        let lease_b = actor_b.lease().expect("lease B");
        let rollout_b = sample_rollout(&lease_b, 13, 1);
        lease_b.try_submit(rollout_b).expect("submit B");
        release.wait();
        assert_eq!(update.join().expect("learner thread").samples_optimized, 1);
    });

    assert_eq!(
        pipeline
            .try_accept()
            .expect("accept concurrent B")
            .rollout_version(),
        RolloutVersion::new(0)
    );
}

#[test]
fn publication_snapshots_only_after_guarded_learner_update_completes() {
    let learner = PolicyModel::fresh(17_029).expect("learner");
    let mut pipeline = ActorLearnerPipeline::new(1, 1, &learner).expect("pipeline");
    let actor = pipeline.take_actor(0).expect("actor");
    let lease = actor.lease().expect("lease");
    let rollout = sample_rollout(&lease, 14, 1);
    lease.try_submit(rollout).expect("submit");
    let batch = pipeline
        .try_accept()
        .expect("accept")
        .finish(single_sample_config())
        .expect("batch");
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let publisher_started = Arc::new(Barrier::new(2));
    let mut trainer = PpoTrainer::new(&learner, single_sample_config(), 17_030).expect("trainer");
    let (published_sender, published_receiver) = std::sync::mpsc::sync_channel(1);

    thread::scope(|scope| {
        let update = scope.spawn(|| {
            trainer
                .train_pipeline_update_with_barriers_for_test(&learner, &batch, &entered, &release)
                .expect("update")
        });
        entered.wait();
        let publisher_started_thread = Arc::clone(&publisher_started);
        let learner_ref = &learner;
        let pipeline_ref = &mut pipeline;
        let publisher = scope.spawn(move || {
            publisher_started_thread.wait();
            let version = pipeline_ref.publish(learner_ref).expect("publish");
            published_sender.send(version).expect("published signal");
        });
        publisher_started.wait();
        assert!(published_receiver.try_recv().is_err());
        release.wait();
        update.join().expect("update thread");
        publisher.join().expect("publisher thread");
    });

    assert_eq!(
        published_receiver.recv().expect("published version"),
        RolloutVersion::new(1)
    );
    assert_eq!(
        actor
            .lease()
            .expect("published lease")
            .policy()
            .export_parameters()
            .expect("actor parameters"),
        learner.export_parameters().expect("learner parameters")
    );
}

#[test]
fn rollout_two_policy_generations_behind_is_rejected() {
    let learner = PolicyModel::fresh(17_030).expect("learner");
    let mut pipeline = ActorLearnerPipeline::new(1, 1, &learner).expect("pipeline");
    let actor = pipeline.take_actor(0).expect("actor");
    let lease = actor.lease().expect("lease");
    let rollout = sample_rollout(&lease, 3, 1);
    lease.try_submit(rollout).expect("submit version zero");
    advance_parameters(&learner);
    pipeline.publish(&learner).expect("publish one");
    advance_parameters(&learner);
    pipeline.publish(&learner).expect("publish two");

    let Err(error) = pipeline.try_accept() else {
        panic!("two-generation stale rollout was accepted");
    };
    assert_eq!(
        error,
        PipelineError::StaleRollout {
            rollout: RolloutVersion::new(0),
            learner: RolloutVersion::new(2),
        }
    );
}

#[test]
fn publication_race_keeps_version_bound_to_immutable_actor_weights() {
    let learner = PolicyModel::fresh(17_040).expect("learner");
    let initial = learner.policy_identity().expect("initial identity");
    let mut pipeline = ActorLearnerPipeline::new(1, 1, &learner).expect("pipeline");
    let actor = pipeline.take_actor(0).expect("actor");
    let leased = Arc::new(Barrier::new(2));
    let published = Arc::new(Barrier::new(2));
    let actor_leased = Arc::clone(&leased);
    let actor_published = Arc::clone(&published);
    let worker = thread::spawn(move || {
        let lease = actor.lease().expect("old lease");
        actor_leased.wait();
        actor_published.wait();
        let rollout = sample_rollout(&lease, 4, 1);
        lease
            .try_submit(rollout)
            .expect("submit old immutable policy");
    });
    leased.wait();
    advance_parameters(&learner);
    pipeline.publish(&learner).expect("publish new policy");
    published.wait();
    worker.join().expect("actor worker");

    let batch = pipeline
        .try_accept()
        .expect("accepted old generation")
        .finish(single_sample_config())
        .expect("batch");
    assert_eq!(batch.rollout_version(), RolloutVersion::new(0));
    assert_eq!(batch.batch().policy(), initial);
}

#[test]
fn pipeline_trainer_accepts_one_generation_despite_multiple_parameter_revisions() {
    let learner = PolicyModel::fresh(17_050).expect("learner");
    let mut pipeline = ActorLearnerPipeline::new(1, 1, &learner).expect("pipeline");
    let actor = pipeline.take_actor(0).expect("actor");
    let lease = actor.lease().expect("lease");
    let rollout = sample_rollout(&lease, 5, 1);
    lease.try_submit(rollout).expect("submit old rollout");
    advance_parameters(&learner);
    advance_parameters(&learner);
    pipeline.publish(&learner).expect("one generation");
    let batch = pipeline
        .try_accept()
        .expect("accepted rollout")
        .finish(single_sample_config())
        .expect("batch");
    let mut trainer = PpoTrainer::new(&learner, single_sample_config(), 17_051).expect("trainer");

    let report = trainer
        .train_pipeline_update(&learner, &batch)
        .expect("pipeline update");

    assert_eq!(report.samples_optimized, 1);
    assert_eq!(report.optimizer_step, 1);
}

#[test]
fn pipeline_trainer_rechecks_live_generation_before_optimizer_mutation() {
    let learner = PolicyModel::fresh(17_055).expect("learner");
    let mut pipeline = ActorLearnerPipeline::new(1, 1, &learner).expect("pipeline");
    let actor = pipeline.take_actor(0).expect("actor");
    let lease = actor.lease().expect("lease");
    let rollout = sample_rollout(&lease, 6, 1);
    lease.try_submit(rollout).expect("submit");
    let batch = pipeline
        .try_accept()
        .expect("accept at generation zero")
        .finish(single_sample_config())
        .expect("batch");
    advance_parameters(&learner);
    pipeline.publish(&learner).expect("generation one");
    advance_parameters(&learner);
    pipeline.publish(&learner).expect("generation two");
    let mut trainer = PpoTrainer::new(&learner, single_sample_config(), 17_056).expect("trainer");
    let before = learner.export_parameters().expect("before");

    let error = trainer
        .train_pipeline_update(&learner, &batch)
        .expect_err("live generation is stale");

    assert_eq!(
        error.to_string(),
        "PPO model error: actor rollout version 0 is stale for learner version 2"
    );
    assert_eq!(learner.export_parameters().expect("after"), before);
    assert_eq!(trainer.optimizer_step(), 0);
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_pipeline_runs_complete_ppo_update_with_cpu_actor_snapshot() {
    accelerator_pipeline_update(PolicyDevice::Cuda { ordinal: 0 });
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "requires a Metal device"]
fn metal_pipeline_runs_complete_ppo_update_with_cpu_actor_snapshot() {
    accelerator_pipeline_update(PolicyDevice::Metal { ordinal: 0 });
}

#[cfg(any(
    all(feature = "cuda", any(target_os = "linux", target_os = "windows")),
    all(feature = "metal", target_os = "macos")
))]
fn accelerator_pipeline_update(device: PolicyDevice) {
    let learner = PolicyModel::fresh_on(17_060, device).expect("accelerator learner");
    let mut pipeline = ActorLearnerPipeline::new(1, 1, &learner).expect("pipeline");
    let actor = pipeline.take_actor(0).expect("CPU actor");
    let lease = actor.lease().expect("lease");
    assert_eq!(lease.policy().device(), PolicyDevice::Cpu);
    let rollout = sample_rollout(&lease, 17_061, 1);
    lease.try_submit(rollout).expect("submit");
    let batch = pipeline
        .try_accept()
        .expect("accept")
        .finish(single_sample_config())
        .expect("batch");
    let mut trainer = PpoTrainer::new(&learner, single_sample_config(), 17_062).expect("trainer");

    let report = trainer
        .train_pipeline_update(&learner, &batch)
        .expect("accelerator PPO update");

    assert_eq!(report.samples_optimized, 1);
    assert_eq!(report.optimizer_step, 1);
}

fn sample_rollout(lease: &ActorPolicyLease, seed: u64, samples: usize) -> ActorRolloutBuffer {
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 10));
    let space = crate::ActionSpace::from_tracker(&tracker).expect("space");
    let frame = encode(&tracker, &LocalPolicyState::new(0));
    let mut rng = PpoRng::new(seed);
    let mut rollout = lease.rollout().expect("rollout");
    for stream in 0..samples {
        let choice = lease
            .policy()
            .sample(&frame, &space, &mut rng)
            .expect("sample");
        rollout
            .push(
                choice
                    .finish(PpoOutcome {
                        stream,
                        decision: 0,
                        ticks: 3,
                        next_value: 0.0,
                        reward: 1.0,
                        terminal: true,
                    })
                    .expect("transition"),
            )
            .expect("push");
    }
    rollout
}

fn advance_parameters(model: &PolicyModel) {
    let parameters = model.export_parameters().expect("parameters");
    model
        .import_parameters(&parameters)
        .expect("advance revision");
}

fn single_sample_config() -> PpoConfig {
    PpoConfig {
        rollout_decisions: 1,
        environments: 1,
        epochs: 1,
        minibatch: 1,
        ..PpoConfig::default()
    }
}
