#[test]
fn cli_rejects_hero_selector() {
    let error = crate::cli::parse_from(["drysua", "--hero", "2"])
        .expect_err("drysua must not accept a hero selector");

    assert!(error.to_string().contains("unexpected argument '--hero'"));
}

#[test]
fn shadow_fiend_pick_is_hero_two() {
    assert_eq!(crate::SHADOW_FIEND, bota_proto::HeroId(2));
}

#[test]
fn cli_accepts_bounded_ppo_smoke_parameters() {
    crate::cli::parse_from([
        "drysua",
        "train",
        "--updates",
        "1",
        "--environments",
        "2",
        "--rollout",
        "8",
        "--epochs",
        "1",
        "--minibatch",
        "16",
        "--seed",
        "77",
        "--map",
        "1",
        "--device",
        "cuda",
        "--device-ordinal",
        "1",
    ])
    .expect("train CLI");
}

#[test]
fn cli_accepts_bounded_self_play_smoke_parameters() {
    crate::cli::parse_from([
        "drysua",
        "league",
        "--updates",
        "1",
        "--environments",
        "4",
        "--rollout",
        "2",
        "--epochs",
        "1",
        "--minibatch",
        "8",
        "--evaluation-pairs",
        "1",
        "--evaluation-decisions",
        "2",
        "--seed",
        "77",
        "--map",
        "1",
        "--device",
        "metal",
        "--device-ordinal",
        "0",
    ])
    .expect("league CLI");
}

#[test]
fn cli_accepts_resumable_training_job_parameters() {
    crate::cli::parse_from([
        "drysua",
        "train-full",
        "--updates",
        "10000",
        "--environments",
        "4",
        "--rollout",
        "8",
        "--epochs",
        "1",
        "--minibatch",
        "32",
        "--checkpoint-seconds",
        "300",
        "--checkpoint-directory",
        "artifacts/training",
        "--resume",
        "--migrate-provenance",
        "--device",
        "cuda",
    ])
    .expect("resumable train CLI");
}

#[test]
fn cli_rejects_provenance_migration_without_resume() {
    let error = crate::cli::parse_from([
        "drysua",
        "train-full",
        "--updates",
        "10000",
        "--checkpoint-directory",
        "artifacts/training",
        "--migrate-provenance",
    ])
    .expect_err("migration requires resume");

    assert!(error.to_string().contains("--resume"));
}

#[test]
fn cli_accepts_initial_weights_for_fresh_training() {
    crate::cli::parse_from([
        "drysua",
        "train-full",
        "--updates",
        "8",
        "--checkpoint-directory",
        "training/ppo-v1",
        "--initial-weights",
        "training/pretrain-v1",
    ])
    .expect("fresh initialized training CLI");
}

#[test]
fn cli_rejects_initial_weights_when_resuming() {
    let error = crate::cli::parse_from([
        "drysua",
        "train-full",
        "--updates",
        "8",
        "--checkpoint-directory",
        "training/ppo-v1",
        "--initial-weights",
        "training/pretrain-v1",
        "--resume",
    ])
    .expect_err("resume already restores exact model state");

    assert!(error.to_string().contains("cannot be used with '--resume'"));
}

#[test]
fn cli_accepts_fixed_checkpoint_evaluation_matrix() {
    crate::cli::parse_from([
        "drysua",
        "evaluate",
        "--checkpoint-directory",
        "training/run/checkpoint",
        "--pairs",
        "2",
        "--decisions",
        "1024",
        "--seed",
        "77",
    ])
    .expect("evaluate CLI");
}

#[test]
fn cli_rejects_partial_checkpoint_evaluation_matrix() {
    let error = crate::cli::parse_from([
        "drysua",
        "evaluate",
        "--checkpoint-directory",
        "training/run/checkpoint",
        "--map",
        "1",
    ])
    .expect_err("evaluation matrix must include both maps");

    assert!(error.to_string().contains("unexpected argument '--map'"));
}

#[test]
fn cli_accepts_bounded_teacher_pretraining() {
    crate::cli::parse_from([
        "drysua",
        "pretrain",
        "--output-directory",
        "training/pretrain-maps",
        "--epochs",
        "8",
        "--seed",
        "50001",
        "--device",
        "cuda",
    ])
    .expect("pretrain CLI");
}

#[test]
fn cli_rejects_partial_map_teacher_pretraining() {
    let error = crate::cli::parse_from([
        "drysua",
        "pretrain",
        "--output-directory",
        "training/pretrain-map1",
        "--map",
        "1",
    ])
    .expect_err("pretraining must cover both maps");

    assert!(error.to_string().contains("unexpected argument '--map'"));
}
