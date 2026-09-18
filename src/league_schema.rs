use crate::{
    ACTION_SCHEMA_HASH, ACTION_SCHEMA_VERSION, FEATURE_SCHEMA_HASH, FEATURE_SCHEMA_VERSION,
    MODEL_SCHEMA_HASH, MODEL_SCHEMA_VERSION, PPO_SCHEMA_HASH, PPO_SCHEMA_VERSION,
};

/// Stage-ten league contract version.
pub const LEAGUE_SCHEMA_VERSION: u32 = 37;
/// Audited simulator and learner rules required by stage-ten league artifacts.
pub const LEAGUE_RULES_AUDIT_VERSION: u32 = 32;
/// Canonical stage-ten frozen-policy, scheduling, retention, and promotion contract.
pub const LEAGUE_SCHEMA_DESCRIPTOR: &str = concat!(
    "bota-drysua-league/v37;",
    "linked_schemas=action,feature,model,ppo,map2_reward;linked_hash=fnv1a_descriptor_then_ordered_version_le32_hash_le64_then_map2_reward_descriptor_utf8;rules_audit=32;",
    "scope=map2_mid_only_second_hero_death_or_first_tower_loss_simultaneous_draw_cap27900_including900_pregame;reward=linked_map2_reward_schema_version_hash_and_full_descriptor;observations=Guarded13_Inspired14_Shadowraze15_Healed_hp_mana_manual_reports_not_confirmed_tickregen;",
    "execution_roles=current_m22_policy_sharedpolicy_current_accepted_new_run_historical_snapshots_use_candidate_feature20_action5_move_landing_order_contract_including_restart,frozen_weights_not_legacy_execution,observer_bookkeeping_never_changes_teacher_strategy;initialization=explicit_pinned_m14_m16_m17_or_m19u162_global_input_padding_or_m21u300_parameter_only_sources_new_map2_m22_policy_not_historical_opponent_identity_or_qualified_release,no_inherited_progress_mastery_moments_rng_league_or_promotion_evidence,no_gameplay_or_reward_equivalence;reward7=terminal_win.2_loss_neg.2_draw0_taskcap_neg.2_win_only_victory_time_bonus_all_dense_unchanged,no_strategy_override;historical_opponents=original_binaries_remain_historical_explicit_map2_adapters_not_original_release_identity,no_old_runtime_or_resume,no_parameter_fingerprint_as_execution_identity;",
    "opponents=current30,accepted25,historical25,teacher15,weak5,frozen_per_rollout;",
    "league=capacity32,minimum9,protect_anchor_accepted_strongest_recent4,evict_nearest_cross_play_profile;",
    "snapshot=immutable_finite_f32_parameters,stable_parameter_fingerprint,generation;",
    "evaluation=held_out_seed_disjoint,paired_radiant_and_dire,min20,max512,authoritative_map2_win_required,draw_and_task_timecap_nonwins,infrastructure_failures_invalidate,timeout_rejected,min_actions1000,rejections_below0.001,weak_loss_and_stall_rejected;",
    "exploit_audit=separate_seed_namespace,min2_pairs,min100_actions,timeout_rejected,nonnegative_each_side,rejections_below0.001;",
    "promotion=opaque_paired_evidence_and_exploit_audit,positive_combined_score,nonnegative_each_side,training_reward_excluded;",
    "current_contract=feature22_model24_ppo37_reward7_win.2_loss_neg.2_draw0_completed_taskcap_neg.2_win_only_victory_time_bonus;mastery_training_gate_is_not_league_qualification;explicit_pinned_parameter_initialization_fresh_mastery_progress_no_equivalence;"
);

/// FNV-1a of the descriptor, ordered linked identities, and reward descriptor.
pub const LEAGUE_SCHEMA_HASH: u64 = crate::model::linked_schema_hash(
    LEAGUE_SCHEMA_DESCRIPTOR,
    &[
        (ACTION_SCHEMA_VERSION, ACTION_SCHEMA_HASH),
        (FEATURE_SCHEMA_VERSION, FEATURE_SCHEMA_HASH),
        (MODEL_SCHEMA_VERSION, MODEL_SCHEMA_HASH),
        (PPO_SCHEMA_VERSION, PPO_SCHEMA_HASH),
        (
            crate::MAP2_REWARD_SCHEMA_VERSION,
            crate::MAP2_REWARD_SCHEMA_HASH,
        ),
    ],
);

const _: () = assert!(ACTION_SCHEMA_VERSION == 5);
const _: () = assert!(FEATURE_SCHEMA_VERSION == 22);
const _: () = assert!(MODEL_SCHEMA_VERSION == 24);
const _: () = assert!(PPO_SCHEMA_VERSION == 37);
const _: () = assert!(LEAGUE_RULES_AUDIT_VERSION == crate::PPO_RULES_AUDIT_VERSION);
