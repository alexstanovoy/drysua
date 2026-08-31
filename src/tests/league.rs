#![allow(
    clippy::float_arithmetic,
    reason = "league evaluation tests use floating-point scores"
)]

use crate::{
    CrossPlayProfile, LEAGUE_SCHEMA_DESCRIPTOR, LEAGUE_SCHEMA_HASH, LEAGUE_SCHEMA_VERSION, League,
    LeagueEvaluation, LeagueExploitAudit, LeagueMatchResult, LeaguePairedResult,
    LeaguePromotionDecision, PolicyModel, PolicySnapshot,
};

#[test]
fn stage_ten_schema_is_stable_and_names_its_safety_contracts() {
    assert_eq!(LEAGUE_SCHEMA_VERSION, 1);
    assert_eq!(LEAGUE_SCHEMA_HASH, 8_903_926_055_252_199_993);
    assert!(LEAGUE_SCHEMA_DESCRIPTOR.contains("held_out_seed_disjoint"));
    assert!(LEAGUE_SCHEMA_DESCRIPTOR.contains("training_reward_excluded"));
}

#[test]
fn opponent_distribution_has_exact_stage_ten_bucket_weights() {
    let model = PolicyModel::fresh(700).expect("model");
    let current = PolicySnapshot::capture(&model, 0).expect("current");
    let league = League::new(9, current.clone()).expect("league");
    let mut counts = [0usize; 5];

    for bucket in 0..100u8 {
        let opponent = league.select_bucket(bucket, &current).expect("opponent");
        counts[opponent.kind().index()] += 1;
    }

    assert_eq!(counts, [30, 25, 25, 15, 5]);
}

#[test]
fn promotion_requires_paired_held_out_sides_and_safe_rejection_rate() {
    let model = PolicyModel::fresh(701).expect("model");
    let accepted = PolicySnapshot::capture(&model, 0).expect("accepted");
    let mut league = League::new(9, accepted.clone()).expect("league");
    let candidate = distinct_snapshot(&model, 1);
    let passing = paired_results(10, LeagueMatchResult::Win, LeagueMatchResult::Win);
    let profile = CrossPlayProfile::new(0.5, 1.0, 0.25).expect("profile");
    let exploit_pairs = paired_results(100, LeagueMatchResult::Draw, LeagueMatchResult::Draw)
        .into_iter()
        .take(2)
        .collect::<Vec<_>>();
    let exploit = LeagueExploitAudit::new(
        candidate.fingerprint(),
        accepted.fingerprint(),
        &exploit_pairs,
        &[1, 2, 3],
        &(10..30).collect::<Vec<_>>(),
        0,
        100,
    )
    .expect("exploit audit");

    let failed = LeagueEvaluation::new(
        candidate.fingerprint(),
        accepted.fingerprint(),
        passing.clone(),
        &[1, 2, 3],
        2,
        2_000,
        profile,
        exploit,
    );
    assert!(failed.is_err(), "0.1% rejection rate must fail");

    let evidence = LeagueEvaluation::new(
        candidate.fingerprint(),
        accepted.fingerprint(),
        passing,
        &[1, 2, 3],
        0,
        2_000,
        profile,
        LeagueExploitAudit::new(
            candidate.fingerprint(),
            accepted.fingerprint(),
            &exploit_pairs,
            &[1, 2, 3],
            &(10..30).collect::<Vec<_>>(),
            0,
            100,
        )
        .expect("exploit audit"),
    )
    .expect("evidence");
    let decision = league
        .try_promote(candidate.clone(), evidence)
        .expect("promotion");

    assert_eq!(decision, LeaguePromotionDecision::Accepted);
    assert_eq!(league.accepted().fingerprint(), candidate.fingerprint());
    assert_eq!(
        league
            .iter()
            .find(|entry| entry.generation() == 1)
            .expect("candidate")
            .profile(),
        profile
    );
}

#[test]
fn historical_checkpoint_does_not_replace_accepted_policy() {
    let model = PolicyModel::fresh(7_011).expect("model");
    let accepted = PolicySnapshot::capture(&model, 0).expect("accepted");
    let accepted_fingerprint = accepted.fingerprint();
    let mut league = League::new(9, accepted).expect("league");

    league
        .insert_historical(
            distinct_snapshot(&model, 1),
            1.0,
            CrossPlayProfile::default(),
        )
        .expect("historical checkpoint");

    assert_eq!(league.accepted().fingerprint(), accepted_fingerprint);
    assert_eq!(league.len(), 2);
}

#[test]
fn historical_bucket_excludes_current_weights_when_an_older_policy_exists() {
    let model = PolicyModel::fresh(7_012).expect("model");
    let accepted = PolicySnapshot::capture(&model, 0).expect("accepted");
    let mut league = League::new(9, accepted).expect("league");
    let current = distinct_snapshot(&model, 1);
    let older = distinct_snapshot(&model, 2);
    league
        .insert_historical(current.clone(), 0.0, CrossPlayProfile::default())
        .expect("current checkpoint");
    league
        .insert_historical(older.clone(), 0.0, CrossPlayProfile::default())
        .expect("older checkpoint");

    let opponent = league.select_bucket(55, &current).expect("historical");

    assert_eq!(
        opponent.snapshot().expect("snapshot").fingerprint(),
        older.fingerprint()
    );
}

#[test]
fn minimum_capacity_can_evict_when_all_retention_roles_are_distinct() {
    let model = PolicyModel::fresh(7_014).expect("model");
    let anchor = PolicySnapshot::capture(&model, 0).expect("anchor");
    let mut league = League::new(9, anchor.clone()).expect("league");
    let accepted = distinct_snapshot(&model, 1);
    league
        .try_promote(
            accepted.clone(),
            passing_evidence(&accepted, &anchor, CrossPlayProfile::default()),
        )
        .expect("accepted checkpoint");
    for generation in 2..=9 {
        let profile = if generation == 2 {
            CrossPlayProfile::new(1.0, -1.0, 1.0).expect("diverse profile")
        } else {
            CrossPlayProfile::default()
        };
        league
            .insert_historical(
                distinct_snapshot(&model, generation),
                generation as f64,
                profile,
            )
            .expect("bounded insertion");
    }

    assert_eq!(league.len(), 9);
    assert_eq!(league.accepted().fingerprint(), accepted.fingerprint());
}

#[test]
fn exploit_audit_rejects_a_regression_on_either_side() {
    let model = PolicyModel::fresh(7_015).expect("model");
    let accepted = PolicySnapshot::capture(&model, 0).expect("accepted");
    let candidate = distinct_snapshot(&model, 1);
    let pairs = vec![
        LeaguePairedResult {
            seed: 100,
            candidate_radiant: LeagueMatchResult::Loss,
            candidate_dire: LeagueMatchResult::Draw,
        },
        LeaguePairedResult {
            seed: 101,
            candidate_radiant: LeagueMatchResult::Draw,
            candidate_dire: LeagueMatchResult::Draw,
        },
    ];

    let error = LeagueExploitAudit::new(
        candidate.fingerprint(),
        accepted.fingerprint(),
        &pairs,
        &[1, 2],
        &[10, 11],
        0,
        100,
    )
    .err()
    .expect("side regression");

    assert_eq!(
        error.to_string(),
        "exploit audit regresses on at least one side"
    );
}

#[test]
fn frozen_snapshot_is_unchanged_after_live_model_parameters_change() {
    let model = PolicyModel::fresh(7_013).expect("model");
    let snapshot = PolicySnapshot::capture(&model, 0).expect("snapshot");
    let frozen_parameters = snapshot.parameters().to_vec();
    let mut changed = model.export_parameters().expect("live parameters");
    changed[0] += 1.0e-4;

    model
        .import_parameters(&changed)
        .expect("change live model");
    let instantiated = snapshot.instantiate().expect("frozen model");

    assert_eq!(snapshot.parameters(), frozen_parameters);
    assert_eq!(
        instantiated.export_parameters().expect("frozen parameters"),
        frozen_parameters
    );
    assert_ne!(
        model.export_parameters().expect("changed parameters"),
        frozen_parameters
    );
}

#[test]
fn bounded_league_keeps_anchor_accepted_strongest_and_recent_entries() {
    let model = PolicyModel::fresh(702).expect("model");
    let anchor = PolicySnapshot::capture(&model, 0).expect("anchor");
    let anchor_fingerprint = anchor.fingerprint();
    let mut league = League::new(9, anchor).expect("league");
    for generation in 1..=12u64 {
        let snapshot = distinct_snapshot(&model, generation);
        league
            .insert_evaluated_for_test(
                snapshot,
                generation as f64,
                CrossPlayProfile::new(
                    generation as f32 / 12.0,
                    -(generation as f32) / 12.0,
                    (generation % 3) as f32 / 2.0,
                )
                .expect("profile"),
            )
            .expect("insert");
    }

    assert_eq!(league.len(), 9);
    assert!(league.contains(anchor_fingerprint), "old anchor retained");
    assert!(league.contains(league.accepted().fingerprint()));
    assert_eq!(league.strongest().score(), 12.0);
    for generation in 9..=12u64 {
        assert!(
            league.iter().any(|entry| entry.generation() == generation),
            "recent generation {generation} retained"
        );
    }
}

#[cfg(feature = "builtin")]
#[test]
fn self_play_smoke_trains_against_scheduled_frozen_opponents_and_pairs_sides() {
    let report = crate::run_league_smoke(crate::LeagueSmokeConfig {
        updates: 1,
        environments: 4,
        rollout_decisions: 2,
        epochs: 1,
        minibatch: 8,
        evaluation_pairs: 1,
        evaluation_decisions: 2,
        seed: 8_811,
        map: bota_proto::MapId(1),
    })
    .expect("league smoke");

    assert_eq!(report.ppo.updates, 1);
    assert_eq!(report.ppo.transitions, 8);
    assert_eq!(report.opponent_counts.iter().sum::<u32>(), 4);
    assert_eq!(report.paired_evaluations, 1);
    assert_eq!(report.profile_evaluations, 2);
    assert_eq!(report.league_policies, 2);
    assert_eq!(report.promotions, 0);
    assert_eq!(report.accepted_after, report.accepted_before);
}

#[cfg(feature = "builtin")]
#[test]
fn promotion_gate_treats_nonterminal_evaluation_horizons_as_draws() {
    let report = crate::run_league_smoke(crate::LeagueSmokeConfig {
        updates: 1,
        environments: 1,
        rollout_decisions: 1,
        epochs: 1,
        minibatch: 1,
        evaluation_pairs: 20,
        evaluation_decisions: 25,
        seed: 8_812,
        map: bota_proto::MapId(1),
    })
    .expect("promotion-gate smoke");

    assert_eq!(report.evaluation_actions, 1_300);
    assert_eq!(report.profile_evaluations, 4);
    assert_eq!(report.exploit_evaluations, 2);
    assert_eq!(report.promotions, 0);
    assert_eq!(report.accepted_after, report.accepted_before);
}

fn paired_results(
    first_seed: u64,
    radiant: LeagueMatchResult,
    dire: LeagueMatchResult,
) -> Vec<LeaguePairedResult> {
    (0..20)
        .map(|offset| LeaguePairedResult {
            seed: first_seed + offset,
            candidate_radiant: radiant,
            candidate_dire: dire,
        })
        .collect()
}

fn distinct_snapshot(model: &PolicyModel, generation: u64) -> PolicySnapshot {
    let mut parameters = model.export_parameters().expect("parameters");
    parameters[0] += generation as f32 * 1.0e-5;
    PolicySnapshot::from_parameters_for_test(parameters, generation).expect("snapshot")
}

fn passing_evidence(
    candidate: &PolicySnapshot,
    accepted: &PolicySnapshot,
    profile: CrossPlayProfile,
) -> LeagueEvaluation {
    let promotion = paired_results(10, LeagueMatchResult::Win, LeagueMatchResult::Win);
    let exploit = paired_results(100, LeagueMatchResult::Draw, LeagueMatchResult::Draw)
        .into_iter()
        .take(2)
        .collect::<Vec<_>>();
    LeagueEvaluation::new(
        candidate.fingerprint(),
        accepted.fingerprint(),
        promotion,
        &[1, 2, 3],
        0,
        2_000,
        profile,
        LeagueExploitAudit::new(
            candidate.fingerprint(),
            accepted.fingerprint(),
            &exploit,
            &[1, 2, 3],
            &(10..30).collect::<Vec<_>>(),
            0,
            100,
        )
        .expect("exploit audit"),
    )
    .expect("promotion evidence")
}
