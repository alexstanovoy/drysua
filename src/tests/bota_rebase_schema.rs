use super::map2_checkpoint::{Directory, runtime_bytes};
use crate::{CheckpointError, PolicyModel, TrainingArtifact};
use std::collections::HashMap;

#[test]
fn rebase_inference_versions_reject_old_effect_and_healing_interpretations() {
    crate::tests::support::assert_frozen_schema_versions();
    assert_eq!(crate::LEAGUE_SCHEMA_VERSION, 37);
    assert_eq!(crate::CHECKPOINT_SCHEMA_VERSION, 12);
    assert_eq!(crate::GLOBAL_FEATURES, 92);
    assert_eq!(crate::UNIT_FEATURES, 84);
    assert!(crate::PPO_SCHEMA_DESCRIPTOR.contains("cap27900"));
    assert!(crate::LEAGUE_SCHEMA_DESCRIPTOR.contains("cap27900"));
}

#[test]
fn rebase_m15_runtime_rejects_before_mutating_parameters_identity_or_optimizer() {
    let model = PolicyModel::fresh(37).expect("model");
    let trainer = crate::PpoTrainer::new(&model, crate::PpoConfig::default(), 38).expect("trainer");
    let prior = model.export_parameters().expect("parameters");
    let identity = model.policy_identity().expect("identity");
    let directory = Directory::new();
    let bytes = runtime_bytes(&vec![0.0; 1_695_924], m15_metadata());
    std::fs::write(directory.0.join("drysua.weights.safetensors"), &bytes).expect("synthetic M15");

    let error = TrainingArtifact::load_runtime_weights(&model, &directory.0)
        .expect_err("old effect schema");

    assert_eq!(error, CheckpointError::SchemaMismatch);
    assert_eq!(
        error.to_string(),
        "checkpoint schema does not match this build"
    );
    assert_eq!(model.export_parameters().expect("parameters"), prior);
    assert_eq!(model.policy_identity().expect("identity"), identity);
    assert!(trainer.checkpoint_snapshot(&model).is_ok());
    assert_eq!(
        std::fs::read(directory.0.join("drysua.weights.safetensors")).expect("source"),
        bytes
    );
}

#[test]
fn rebase_m15_shape_cannot_be_relabelled_as_current_runtime() {
    let directory = Directory::new();
    let model = PolicyModel::fresh(40).expect("model");
    TrainingArtifact::save_runtime_weights(&model, &directory.0).expect("synthetic current");
    let current =
        std::fs::read(directory.0.join("drysua.weights.safetensors")).expect("current bytes");
    let (_, header) = safetensors::SafeTensors::read_metadata(&current).expect("metadata");
    let bytes = runtime_bytes(
        &vec![0.0; 1_695_924],
        header.metadata().clone().expect("metadata"),
    );
    std::fs::write(directory.0.join("drysua.weights.safetensors"), bytes).expect("wrong shape");
    let identity = model.policy_identity().expect("identity");

    let error =
        TrainingArtifact::load_runtime_weights(&model, &directory.0).expect_err("no relabel");

    assert_eq!(
        error.to_string(),
        "checkpoint tensor contract has invalid dtype or shape"
    );
    assert_eq!(model.policy_identity().expect("identity"), identity);
}

#[test]
fn rebase_checkpoint3_rejects_before_reading_any_tensor_file() {
    let directory = Directory::new();
    let mut bytes = b"DRYCKP18".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend(6_904_067_705_245_923_052u64.to_le_bytes());
    std::fs::write(directory.0.join("checkpoint.meta"), &bytes).expect("old manifest header");

    let error = TrainingArtifact::load(&directory.0).expect_err("no old resume");

    assert_eq!(error, CheckpointError::SchemaMismatch);
    assert_eq!(
        error.to_string(),
        "checkpoint schema does not match this build"
    );
    assert_eq!(
        std::fs::read(directory.0.join("checkpoint.meta")).expect("unchanged header"),
        bytes
    );
    assert!(!directory.0.join("checkpoint.safetensors").exists());
}

/// Exact old v1 contract for rejection fixtures only.
const MAP2_REWARD_V1_DESCRIPTOR: &str = concat!(
    "drysua-map2-reward/v1;map2_1v1_seat_snapshot_events_contiguous_tick_complete;",
    "units4096_events4096_identities8192_towers64_tick3600000_amount1000000_xp1000000000;",
    "identity_opaque_full_generation_public_scoreboard_heroes_retained_other_metadata480ticks;",
    "snapshot_capacity_preflight_death_structure_current_and_prior_role_validation_no_alive_victim_or_known_resurrection;",
    "gold_observed_paid_died_own_minus_enemy_no_cash_networth_passive_sales_or_lh_double_payment;",
    "xp_public_positive_increments_own_minus_enemy;",
    "hero_damage_positive_reported_mitigated_own_hero_to_opposing_hero_no_creep_damage;",
    "received_own_hero_from_hero_creep_other_unknown_separate_no_healing_reward;",
    "mana_positive_same_body_same_capacity_previous_minus_current_no_request_cost_capacity_change_unobserved;",
    "channels=own_gold:.03/300,enemy_gold:-.03/300,own_xp:.03/3000,enemy_xp:-.03/3000,",
    "hero_dealt:.08/1600,hero_taken:-.025/1600,creep_taken:-.01/500,other_taken:-.005/500,mana:-.04/1200;",
    "channel_payout=budget*scale*amount/((scale+prior_count)*(scale+prior_count+amount));",
    "nonreplenishing_separate_unsigned_counts_state_remaining_scale_over_scale_plus_count;",
    "tower=.05*(mean_own_hp_fraction-mean_enemy_hp_fraction)_public_cached_no_absence_death;",
    "lane=.01*(mean_own_creep_axis+mean_enemy_creep_axis-1)_fountain_axis_public_both_cohorts_else_hold;",
    "potentials_exact_gamma1_deltas_not_budget_clipped_first_tick_resources_potentials_baseline_events_counted;",
    "terminal_win1_loss-1_draw0_timecap0_distinct_lane_zero_tower_final_retained;",
    "gamma1_only_dense_absolute_net_return_bound.4_no_strategy_masks_or_teacher_inputs;"
);

fn m15_metadata() -> HashMap<String, String> {
    [
        ("action_schema_hash", "14080316840523410707".to_owned()),
        ("feature_schema_hash", "612467982395246657".to_owned()),
        ("model_schema_hash", "149485500614302181".to_owned()),
        ("ppo_schema_version", "28".to_owned()),
        ("ppo_schema_hash", "16579842539143021978".to_owned()),
        ("ppo_rules_audit_version", "23".to_owned()),
        ("map2_reward_schema_version", "1".to_owned()),
        ("map2_reward_schema_hash", "798798703797057220".to_owned()),
        (
            "map2_reward_schema_descriptor",
            MAP2_REWARD_V1_DESCRIPTOR.to_owned(),
        ),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_owned(), value))
    .collect()
}
