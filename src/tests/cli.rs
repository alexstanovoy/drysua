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
