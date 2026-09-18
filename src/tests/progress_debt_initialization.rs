use super::map2_checkpoint::{Directory, runtime_bytes};
use crate::tests::support::{assert_bits, assert_fresh_state};
use crate::{CheckpointError, PolicyModel, TrainingArtifact};
use std::collections::HashMap;

#[test]
fn current_progress_debt_shapes_and_action_contract_remain_unchanged() {
    crate::tests::support::assert_frozen_schema_versions();
    assert_eq!(crate::LEAGUE_SCHEMA_VERSION, 37);
    assert_eq!(crate::CHECKPOINT_SCHEMA_VERSION, 12);
    assert_eq!(crate::IMITATION_RULES_AUDIT_VERSION, 22);
    assert_eq!(crate::MODEL_PARAMETER_COUNT, 1_700_020);
    assert_eq!(crate::ACTION_SCHEMA_HASH, 10_658_390_830_565_586_343);
    assert_eq!(crate::MAP2_REWARD_SCHEMA_HASH, 7_274_660_837_025_042_530);
}

#[test]
fn progress_debt_runtime_rejects_exact_m18_reward2_before_mutating_model_or_optimizer() {
    let directory = Directory::new();
    let path = directory.0.join("drysua.weights.safetensors");
    let bytes = runtime_bytes(&vec![0.0; 1_697_460], m18_metadata());
    std::fs::write(&path, &bytes).expect("synthetic previous runtime");
    let model = PolicyModel::fresh(9131900).expect("model");
    let before = model.export_parameters().expect("parameters");
    let identity = model.policy_identity().expect("identity");
    let trainer =
        crate::PpoTrainer::new(&model, super::map2_checkpoint::config(), 1).expect("trainer");
    let error =
        TrainingArtifact::load_runtime_weights(&model, &directory.0).expect_err("old runtime");
    assert_eq!(error, CheckpointError::SchemaMismatch);
    assert_eq!(
        error.to_string(),
        "checkpoint schema does not match this build"
    );
    assert_bits(&model.export_parameters().expect("after"), &before);
    assert_eq!(model.policy_identity().expect("identity"), identity);
    assert_fresh_state(&model, &trainer);
    assert_eq!(std::fs::read(path).expect("source unchanged"), bytes);
}

#[test]
fn progress_debt_resume_rejects_checkpoint6_before_tensor_access() {
    let directory = Directory::new();
    let mut bytes = b"DRYCKP18".to_vec();
    bytes.extend(6u32.to_le_bytes());
    bytes.extend(16_772_919_360_388_607_733u64.to_le_bytes());
    std::fs::write(directory.0.join("checkpoint.meta"), &bytes).expect("old header");
    let error = TrainingArtifact::load(&directory.0).expect_err("old checkpoint");
    assert_eq!(error, CheckpointError::SchemaMismatch);
    assert_eq!(
        error.to_string(),
        "checkpoint schema does not match this build"
    );
    assert_eq!(
        std::fs::read(directory.0.join("checkpoint.meta")).expect("unchanged"),
        bytes
    );
    assert!(!directory.0.join("checkpoint.safetensors").exists());
}

#[test]
fn progress_debt_old_reward2_literal_stays_frozen_without_asserting_a_symmetric_bound() {
    let descriptor = m18_metadata()
        .remove("map2_reward_schema_descriptor")
        .expect("old descriptor");
    let hash = descriptor
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
        });
    assert_eq!(hash, 699_687_995_158_557_285);
    assert_ne!(descriptor, crate::MAP2_REWARD_SCHEMA_DESCRIPTOR);
    assert_eq!(crate::MAP2_REWARD_STAGNATION_MAX_BASE_CHARGES, 8);
}

/// Exact old v2 contract for rejection fixtures only; no M18 source is whitelisted.
const MAP2_REWARD_V2_DESCRIPTOR: &str = concat!(
    "drysua-map2-reward/v2;map2_1v1_seat_snapshot_events_contiguous_tick_complete;",
    "units4096_events4096_identities8192_towers64_tick27900_amount1000000_xp1000000000;",
    "public_metadata=map2_rate30_terrain_axis1to512_pregame0to27900;",
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
    "pregame_movement=.005_times_one_minus_clamped_euclidean_distance_to9216_9216_over9216sqrt2,only_pending_tick_lt_public_pregame_ticks,first_complete_observed_body_baseline_free,missing_body_holds_last_observed_potential,reappearance_uses_observed_position,no_cutoff_or_terminal_reversal,no_postspawn_hero_position_reward;",
    "fountain_wait=own_live_full_projected_hp_mana_both_consecutive_snapshots_same_full_generation_raw_position_inside_observed_own_fountain1200_inclusive,first_eligible_elapsed0,grace30_at30_base.0001_then.00005_per_second_prorated_div30_per_tick,incremental_negative_cost,any_movement_or_condition_break_resets_without_refund;",
    "fountain_purchase=any_confirmed_own_ItemBought_priority_before_condition_break_refunds_entire_open_period_including_drained_charges_then_resets_elapsed0_no_new_wait_interval,enemy_buy_ignored,no_price_intent_channel_saving_exceptions;",
    "state_additions=fountain_wait_ticks_u32_and_current_refundable_cost_f32_only;",
    "terminal_win1_loss-1_draw0_timecap0_distinct_lane_zero_tower_final_retained;",
    "finish_preserves_pregame_hint_and_emitted_wait_total_no_further_wait_charge_or_refund;",
    "gamma1_only_dense_absolute_net_return_bound=.4_v1+.005_center+.0001_times27900over30=.498,wait_rate_le_base_refund_le_charged_current_period_no_wait_clipping_no_strategy_masks_or_teacher_inputs;"
);

fn m18_metadata() -> HashMap<String, String> {
    [
        ("action_schema_hash", "10658390830565586343"),
        ("feature_schema_hash", "17888785275670453418"),
        ("model_schema_hash", "3900982062969752096"),
        ("ppo_schema_version", "31"),
        ("ppo_schema_hash", "15379677344330093698"),
        ("ppo_rules_audit_version", "26"),
        ("map2_reward_schema_version", "2"),
        ("map2_reward_schema_hash", "699687995158557285"),
        ("map2_reward_schema_descriptor", MAP2_REWARD_V2_DESCRIPTOR),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_owned(), value.to_owned()))
    .collect()
}
