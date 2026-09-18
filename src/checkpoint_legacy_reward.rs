#[cfg(test)]
#[path = "tests/checkpoint_legacy_reward_test_support.rs"]
mod test_support;
#[cfg(test)]
pub(crate) use test_support::*;

/// Original reward-v1 metadata for immutable M16/M17 source readers only.
pub(crate) const MAP2_REWARD_V1_DESCRIPTOR: &str = concat!(
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

const _: () = assert!(
    crate::model::fnv1a_extend(
        crate::model::FNV_OFFSET,
        MAP2_REWARD_V1_DESCRIPTOR.as_bytes(),
    ) == 798_798_703_797_057_220
);

/// Frozen reward3 descriptor for explicit M19 initialization, never a current runtime alias.
pub(crate) const MAP2_REWARD_V3_DESCRIPTOR: &str = concat!(
    "drysua-map2-reward/v3;map2_1v1_seat_snapshot_events_contiguous_tick_complete;",
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
    "wait_state=fountain_wait_ticks_u32_current_refundable_cost_f32;",
    "progress_flags=u16_or_per_tick_xp1_gold2_hero_damage4_structure_damage8_creep_kill16_creep_deny32_purchase64_fountain_aura128_pregame_movement256_nearby_wave_pressure512;",
    "progress_sources=own_xp_gain_own_paid_bounty_own_hero_to_enemy_hero_damage_own_hero_to_enemy_tower_barracks_ancient_damage_own_nondenied_creep_kill_including_zero_gold_own_creep_deny_any_confirmed_own_purchase;",
    "progress_snapshot=own_effect3_positive_ticks_even_full_without_regen_requirement_positive_prewave_center_increment_positive_existing_wave_increment_only_with_live_own_hero_within1500_of_visible_own_live_lane_creep;",
    "progress_detection=completed_tick_counter_deltas_before_journal_trim_and_retention_not_accumulated_interval_totals_no_passive_gold_enemy_progress_unknown_targets_clicks_empty_casts_or_other_hero_movement;",
    "progress_debt=baseline_free_all_completed_ticks_including_dead_clamp0to2700_any_reason_refreshes30tick_lease_current_tick_included_no_stacking_active_repay_min3_then_consume1_lease_inactive_add1;",
    "progress_penalty=base.02_at_first2700_no_rate_same_tick_latch_until_debt0_subsequent_inactive_ticks_at2700_cost.000002_partial_repay_preserves_latch_no_refund_no_reward_clipping;",
    "progress_state=stagnation_ticks_u32_activity_ticks_left_u32_stagnation_base_charged_bool_only;progress_purchase=lease_only_never_debt_reset_independent_of_unchanged_v2_fountain_full_refund;",
    "terminal_win1_loss-1_draw0_timecap0_distinct_lane_zero_tower_final_retained;",
    "finish_preserves_pregame_hint_wait_and_stagnation_totals_no_extra_charge_repayment_or_refund;",
    "v2_dense_bound=.4_v1+.005_center+.0001_times27900over30=.498,wait_rate_le_base_refund_le_charged_current_period_no_wait_clipping;",
    "progress_bounds=max_base_charges1plus27900minus2700_over2700plus900=8_cost_bound8times.02_plus27900times.000002=.2158;",
    "gamma1_only_full_episode_negative_absolute_bound.7138_positive_bound.255_from_positive_budgets.14_tower.1_terminal_lane.01_center.005_sum.9688_lt1_no_strategy_masks_or_teacher_inputs;"
);
