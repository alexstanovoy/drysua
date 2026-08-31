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
    ])
    .expect("league CLI");
}
