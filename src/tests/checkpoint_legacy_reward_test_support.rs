use super::*;

/// Exact old v2 contract for rejection fixtures only; no M18 source is whitelisted.
#[cfg(test)]
pub(crate) const MAP2_REWARD_V2_DESCRIPTOR: &str = concat!(
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

#[cfg(test)]
const _: () = assert!(
    crate::model::fnv1a_extend(
        crate::model::FNV_OFFSET,
        MAP2_REWARD_V2_DESCRIPTOR.as_bytes(),
    ) == 699_687_995_158_557_285
);

/// Exact reward4 reference for old M20 rejection fixtures; derived only from frozen reward3.
#[cfg(test)]
pub(crate) fn reward_v4_descriptor() -> String {
    MAP2_REWARD_V3_DESCRIPTOR
        .replacen("drysua-map2-reward/v3;", "drysua-map2-reward/v4;", 1)
        .replacen("terminal_win1_loss-1_draw0_timecap0_distinct_lane_zero_tower_final_retained;",
            "terminal_win1_loss-1_draw-1_timecap-1_distinct_outcome_labels_lane_zero_tower_final_retained;infrastructure_errors_never_terminal_rewards;", 1)
        + "terminal_dominance=win1_minus_nonwin_neg1_minus_negative_dense.7138_minus_positive_dense.255=1.0312_gt0;"
}
