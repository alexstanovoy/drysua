#![allow(
    clippy::float_arithmetic,
    reason = "Seat-only reward accounting uses bounded f64 arithmetic"
)]

mod events;
mod observation;
mod opening;
mod potential;
mod progress;

use std::collections::BTreeMap;
use std::fmt;

use bota_proto::{EntityId, EventKind, MatchInfo, SlotId};
use observation::{Identity, Role, SnapshotFacts};
use potential::Tower;

pub use progress::*;

/// Independent metadata version for the Map2 reward profile.
pub const MAP2_REWARD_SCHEMA_VERSION: u32 = 3;
/// Maximum events consumed atomically in one seat-visible tick.
pub const MAP2_REWARD_MAX_EVENTS: usize = 4096;
/// Maximum visible units accepted in one snapshot.
pub const MAP2_REWARD_MAX_UNITS: usize = 4096;
/// Maximum compact public identity records, including recently absent units.
pub const MAP2_REWARD_MAX_IDENTITIES: usize = 8192;
/// Number of independently diminishing event-credit and cost channels.
pub const MAP2_REWARD_CHANNELS: usize = 9;
/// Required per-tick discount for exact potential cancellation and the return bound.
pub const MAP2_REWARD_GAMMA_TICK: f32 = 1.0;
/// Maximum pre-wave center-distance potential, in reward units.
pub const MAP2_REWARD_PREGAME_CENTER_SCALE: f64 = 0.005;
/// Complete stationary/full fountain intervals before the initial wait charge.
pub const MAP2_REWARD_FOUNTAIN_GRACE_TICKS: u32 = crate::MAP2_TICK_RATE;
/// Initial fountain wait cost at the grace boundary, in reward units.
pub const MAP2_REWARD_FOUNTAIN_BASE_COST: f64 = 0.0001;
/// Fountain wait cost per second after the grace boundary, prorated per tick.
pub const MAP2_REWARD_FOUNTAIN_COST_PER_SECOND: f64 = 0.00005;
/// Upper bound on total wait charges over the native Map2 cap, before refunds.
pub const MAP2_REWARD_FOUNTAIN_WAIT_BOUND: f64 = MAP2_REWARD_FOUNTAIN_BASE_COST
    * crate::MAP2_TICK_CAP as f64
    / MAP2_REWARD_FOUNTAIN_GRACE_TICKS as f64;
/// Unchanged v2 absolute dense bound, excluding stagnation and terminal reward.
pub const MAP2_REWARD_V2_DENSE_BOUND: f64 =
    V1_DENSE_BOUND + MAP2_REWARD_PREGAME_CENTER_SCALE + MAP2_REWARD_FOUNTAIN_WAIT_BOUND;
/// Inactive debt threshold, clamped so sustained activity can always repay it.
pub const MAP2_REWARD_STAGNATION_THRESHOLD_TICKS: u32 = 90 * crate::MAP2_TICK_RATE;
/// Active ticks granted by one useful tick, including the useful tick itself.
pub const MAP2_REWARD_ACTIVITY_LEASE_TICKS: u32 = crate::MAP2_TICK_RATE;
/// Debt ticks repaid per active tick, capped at the current debt.
pub const MAP2_REWARD_STAGNATION_REPAY_PER_TICK: u32 = 3;
/// Base cost charged once per stall bout until its debt is fully repaid.
pub const MAP2_REWARD_STAGNATION_BASE_COST: f64 = 0.02;
/// Cost on subsequent inactive ticks at the threshold while the base is latched.
pub const MAP2_REWARD_STAGNATION_TICK_COST: f64 = 0.000002;
/// Conservative native-cap maximum of fully separated base-charge bouts.
pub const MAP2_REWARD_STAGNATION_MAX_BASE_CHARGES: u32 = 1
    + (MAX_TICK - MAP2_REWARD_STAGNATION_THRESHOLD_TICKS)
        / (MAP2_REWARD_STAGNATION_THRESHOLD_TICKS
            + MAP2_REWARD_STAGNATION_THRESHOLD_TICKS
                .div_ceil(MAP2_REWARD_STAGNATION_REPAY_PER_TICK));
/// Upper bound on stagnation charges; no runtime reward clipping is applied.
pub const MAP2_REWARD_STAGNATION_BOUND: f64 = MAP2_REWARD_STAGNATION_MAX_BASE_CHARGES as f64
    * MAP2_REWARD_STAGNATION_BASE_COST
    + MAX_TICK as f64 * MAP2_REWARD_STAGNATION_TICK_COST;
/// Upper bound on absolute undiscounted dense episode return, excluding terminal reward.
pub const MAP2_REWARD_DENSE_BOUND: f64 = MAP2_REWARD_V2_DENSE_BOUND + MAP2_REWARD_STAGNATION_BOUND;
/// Upper bound on positive dense return over a FULL episode including terminal lane closure.
pub const MAP2_REWARD_POSITIVE_BOUND: f64 = BUDGETS[0]
    + BUDGETS[2]
    + BUDGETS[4]
    + 2.0 * TOWER_SCALE
    + LANE_SCALE
    + MAP2_REWARD_PREGAME_CENTER_SCALE;
/// Exact accounting and calibration contract; coefficients are engineering choices.
pub const MAP2_REWARD_SCHEMA_DESCRIPTOR: &str = concat!(
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
/// Stable FNV-1a hash of the complete independent reward descriptor.
pub const MAP2_REWARD_SCHEMA_HASH: u64 = schema_hash(MAP2_REWARD_SCHEMA_DESCRIPTOR.as_bytes());

const MAX_TICK: u32 = crate::MAP2_TICK_CAP;
const V1_DENSE_BOUND: f64 = 0.4;
const MAX_AMOUNT: i32 = 1_000_000;
const MAX_XP: i32 = 1_000_000_000;
const IDENTITY_AGE: u32 = 480;
const MAX_TOWERS: usize = 64;
const TOWER_SCALE: f64 = 0.05;
const LANE_SCALE: f64 = 0.01;
const BUDGETS: [f64; MAP2_REWARD_CHANNELS] =
    [0.03, 0.03, 0.03, 0.03, 0.08, 0.025, 0.01, 0.005, 0.04];
const SCALES: [f64; MAP2_REWARD_CHANNELS] = [
    300.0, 300.0, 3000.0, 3000.0, 1600.0, 1600.0, 500.0, 500.0, 1200.0,
];
const _: () = assert!(MAP2_REWARD_MAX_IDENTITIES >= 2 * MAP2_REWARD_MAX_UNITS);
const _: () = assert!(
    event_budget_total() + 2.0 * TOWER_SCALE + 2.0 * LANE_SCALE <= V1_DENSE_BOUND + 1.0e-12
);
const _: () = assert!(MAX_TICK == 27_900);
const _: () = assert!(MAP2_REWARD_FOUNTAIN_GRACE_TICKS == 30);
const _: () = assert!(MAP2_REWARD_FOUNTAIN_BASE_COST > 0.0);
const _: () = assert!(MAP2_REWARD_FOUNTAIN_COST_PER_SECOND <= MAP2_REWARD_FOUNTAIN_BASE_COST);
const _: () = assert!(MAP2_REWARD_FOUNTAIN_COST_PER_SECOND > 0.0);
const _: () = assert!(MAP2_REWARD_STAGNATION_THRESHOLD_TICKS == 2700);
const _: () = assert!(MAP2_REWARD_ACTIVITY_LEASE_TICKS == 30);
const _: () = assert!(MAP2_REWARD_STAGNATION_REPAY_PER_TICK > 0);
const _: () = assert!(MAP2_REWARD_STAGNATION_MAX_BASE_CHARGES == 8);
const _: () = assert!(MAP2_REWARD_STAGNATION_BASE_COST > 0.0);
const _: () = assert!(MAP2_REWARD_STAGNATION_TICK_COST > 0.0);
const _: () = assert!(MAP2_REWARD_DENSE_BOUND + MAP2_REWARD_POSITIVE_BOUND < 1.0);
const _: () = assert!(
    (MAX_TICK as u64) * (MAP2_REWARD_MAX_EVENTS as u64) * (MAX_AMOUNT as u64) < u64::MAX / 2
);

/// Authoritative game result or explicit learner task deadline; never a technical failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Map2RewardEnd {
    Win,
    Loss,
    Draw,
    TimeCap,
}

/// Raw seat-visible measurements accumulated over a decision interval.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Map2RewardObservations {
    pub own_gold_earned: u64,
    pub enemy_gold_earned: u64,
    pub own_xp_gained: u64,
    pub enemy_xp_gained: u64,
    pub hero_damage_dealt: u64,
    /// Own hero damage to opposing towers, barracks and Ancients; no separate instant credit.
    pub structure_damage_dealt: u64,
    /// Classified own non-denied creep kills, including zero-paid-gold kills.
    pub creep_kills: u64,
    /// Classified own friendly-creep denies; no separate instant credit.
    pub creep_denies: u64,
    pub hero_damage_taken: u64,
    pub creep_damage_taken: u64,
    /// Damage from known non-hero/non-creep sources or the environment.
    pub other_damage_taken: u64,
    /// Received damage whose nonempty source handle has no public classification.
    pub unattributed_damage_taken: u64,
    pub mana_spent: u64,
    /// Ticks with body replacement/absence or capacity changes that prevent net-spend comparison.
    pub mana_unobserved_ticks: u64,
    /// Own paid, non-denied creep deaths; gold already supplies their reward.
    pub lane_last_hits: u64,
    pub neutral_last_hits: u64,
    /// Positive damage events with an unknown target or nonempty unknown source, once per event.
    pub unattributed_damage_events: u64,
    pub unattributed_deaths: u64,
    pub duplicate_deaths: u64,
    /// Completed nonbaseline ticks with both lane cohorts and a public fountain axis.
    pub lane_observed_ticks: u64,
    /// Consecutive full stationary own-fountain intervals, including grace intervals.
    pub fountain_wait_ticks: u64,
    /// Intervals emitting a wait cost, including each period's base-charge interval.
    pub fountain_wait_charged_ticks: u64,
    /// Own purchase ticks refunding a positive open-period charge, once per tick.
    pub fountain_wait_refunds: u64,
    pub stagnation_active_ticks: u64,
    pub stagnation_idle_ticks: u64,
    /// Inactive ticks charged either the base or subsequent tick cost.
    pub stagnation_charged_ticks: u64,
    pub stagnation_base_charges: u64,
    /// Actual debt removed, not the requested repayment when debt is already small.
    pub stagnation_repaid_ticks: u64,
    /// Bitwise OR of the useful-activity reasons observed during this interval.
    pub progress_reasons: u16,
}

/// Normalized f64 components; `total` is their sum and `ticks` excludes the initial baseline.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Map2RewardBreakdown {
    pub ticks: u32,
    pub gold: f64,
    pub experience: f64,
    pub hero_damage: f64,
    pub hero_damage_taken: f64,
    pub creep_damage_taken: f64,
    pub other_damage_taken: f64,
    /// Negative opportunity cost of observed net mana spend, not a raw mana amount.
    pub mana_spent: f64,
    pub tower_health: f64,
    pub lane_pressure: f64,
    pub pregame_movement: f64,
    pub fountain_wait: f64,
    pub fountain_wait_refund: f64,
    pub stagnation_base: f64,
    pub stagnation_ticks_cost: f64,
    pub terminal: f64,
    pub total: f64,
    pub end: Option<Map2RewardEnd>,
    pub observations: Map2RewardObservations,
}

/// ID-free observable accounting state for a policy/critic input or diagnostic log.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Map2RewardState {
    /// Unspent fraction in descriptor channel order, each in `(0, 1]`.
    pub remaining: [f32; MAP2_REWARD_CHANNELS],
    /// Current bounded public building-health potential, in `[-0.05, 0.05]`.
    pub tower_potential: f32,
    /// Current bounded observed wave potential, in `[-0.01, 0.01]`.
    pub lane_potential: f32,
    pub lane_observed: bool,
    /// Full stationary intervals in the open own-fountain wait period.
    pub fountain_wait_ticks: u32,
    /// Current period's charged cost refundable by any confirmed own purchase.
    pub fountain_wait_refundable_cost: f32,
    /// Progress debt, in 0..=2700 ticks; not reset by death or movement.
    pub stagnation_ticks: u32,
    /// Remaining active intervals, in 0..30 after a completed tick.
    pub activity_ticks_left: u32,
    /// Whether this stall bout's base was charged; cleared only when debt reaches zero.
    pub stagnation_base_charged: bool,
    /// Last fully consumed Snapshot/Events tick; absent before the first complete pair.
    pub completed_tick: Option<u32>,
}

/// A rejected observation or lifecycle call; rejected batches do not mutate committed state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Map2RewardError {
    Invalid(&'static str),
    Limit {
        field: &'static str,
        actual: usize,
        maximum: usize,
    },
}

impl fmt::Display for Map2RewardError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(formatter, "Map2 reward: {message}"),
            Self::Limit {
                field,
                actual,
                maximum,
            } => {
                write!(
                    formatter,
                    "Map2 reward: {field} has {actual} entries; maximum is {maximum}"
                )
            }
        }
    }
}

impl std::error::Error for Map2RewardError {}

/// Complete-tick reward accounting for one Map2 seat; no simulator state or strategic actions.
#[derive(Clone, Debug)]
pub struct Map2Reward {
    roles: [Role; 2],
    pregame_ticks: u32,
    current: Option<SnapshotFacts>,
    pending: Option<SnapshotFacts>,
    identities: BTreeMap<EntityId, Identity>,
    towers: BTreeMap<EntityId, Tower>,
    counts: [u64; MAP2_REWARD_CHANNELS],
    tower_potential: f64,
    lane_potential: f64,
    lane_observed: bool,
    pregame_center_potential: Option<f64>,
    fountain_wait_ticks: u32,
    fountain_wait_refundable_cost: f64,
    stagnation_ticks: u32,
    activity_ticks_left: u32,
    stagnation_base_charged: bool,
    interval: Map2RewardBreakdown,
    ended: bool,
}

impl Map2Reward {
    /// Starts a Map2 1v1 observer; match identity, seeds and private economy are not stored.
    pub fn new(slot: SlotId, info: &MatchInfo) -> Result<Self, Map2RewardError> {
        let roles = observation::roles(slot, info)?;
        assert_ne!(roles[0].team, roles[1].team);
        assert_ne!(roles[0].slot, roles[1].slot);
        Ok(Self {
            roles,
            pregame_ticks: info.pregame_ticks,
            current: None,
            pending: None,
            identities: BTreeMap::new(),
            towers: BTreeMap::new(),
            counts: [0; MAP2_REWARD_CHANNELS],
            tower_potential: 0.0,
            lane_potential: 0.0,
            lane_observed: false,
            pregame_center_potential: None,
            fountain_wait_ticks: 0,
            fountain_wait_refundable_cost: 0.0,
            stagnation_ticks: 0,
            activity_ticks_left: 0,
            stagnation_base_charged: false,
            interval: Map2RewardBreakdown::default(),
            ended: false,
        })
    }

    /// Stages one fogged snapshot; its matching Events must arrive before any further snapshot.
    pub fn observe_snapshot(
        &mut self,
        view: &bota_proto::WorldView,
    ) -> Result<(), Map2RewardError> {
        self.ensure_live()?;
        if self.pending.is_some() {
            return invalid("Snapshot arrived before pending Events");
        }
        if let Some(current) = &self.current
            && current.tick.checked_add(1) != Some(view.tick)
        {
            return invalid("Snapshot ticks must be contiguous");
        }
        let pending = observation::snapshot(view, self.roles)?;
        self.check_snapshot_progress(&pending)?;
        self.check_identity_capacity(&pending)?;
        self.check_tower_capacity(&pending)?;
        assert!(view.tick > 0);
        assert!(view.tick <= MAX_TICK);
        self.pending = Some(pending);
        Ok(())
    }

    /// Validates and consumes every event exactly once before journal eviction or PPO retention.
    pub fn observe_events(
        &mut self,
        tick: u32,
        events: &[EventKind],
    ) -> Result<(), Map2RewardError> {
        self.ensure_live()?;
        let Some(pending) = &self.pending else {
            return invalid("Events without pending Snapshot");
        };
        if pending.tick != tick {
            return invalid("Events tick does not match pending Snapshot");
        }
        events::validate(events)?;
        self.validate_event_lifecycle(pending, events)?;
        let pending = self.pending.take().expect("pending snapshot was validated");
        let previous_interval = self.interval;
        self.update_identities(&pending);
        self.update_towers(&pending);
        self.observe_resources(&pending);
        for event in events {
            self.observe_event(event);
        }
        self.observe_potentials(&pending);
        self.observe_pregame_movement(&pending);
        self.observe_fountain_wait(&pending, events);
        self.observe_progress(&pending, events, previous_interval);
        if self.current.is_some() {
            self.interval.ticks += 1;
        }
        self.current = Some(pending);
        self.interval.retotal();
        assert!(self.identities.len() <= MAP2_REWARD_MAX_IDENTITIES);
        assert!(self.interval.total.is_finite());
        Ok(())
    }

    /// Drains completed interval credit without resetting lifetime budgets, identities or potentials.
    pub fn take_interval(&mut self) -> Result<Map2RewardBreakdown, Map2RewardError> {
        self.ensure_complete()?;
        let result = std::mem::take(&mut self.interval);
        assert!(result.total.is_finite());
        assert!(result.end.is_none());
        Ok(result)
    }

    /// Drains the final interval, closes lane potential, and retains final tower-health progress.
    /// `TimeCap` is a learner terminal, not an invented `MatchOver` or a technical timeout.
    /// Pregame, wait/refund and stagnation totals receive no additional charge or repayment.
    pub fn finish(&mut self, end: Map2RewardEnd) -> Result<Map2RewardBreakdown, Map2RewardError> {
        self.ensure_complete()?;
        self.interval.lane_pressure -= self.lane_potential;
        self.lane_potential = 0.0;
        self.lane_observed = false;
        self.interval.terminal = match end {
            Map2RewardEnd::Win => 1.0,
            Map2RewardEnd::Loss => -1.0,
            Map2RewardEnd::Draw | Map2RewardEnd::TimeCap => 0.0,
        };
        self.interval.end = Some(end);
        self.interval.retotal();
        self.ended = true;
        let result = std::mem::take(&mut self.interval);
        assert!(result.total.is_finite());
        assert!(result.end.is_some());
        Ok(result)
    }

    /// Copies observable budgets, potentials, wait and progress state; no identity handle enters it.
    pub fn state(&self) -> Map2RewardState {
        let remaining = std::array::from_fn(|index| {
            (SCALES[index] / (SCALES[index] + self.counts[index] as f64)) as f32
        });
        assert!(remaining.iter().all(|value| *value > 0.0));
        assert!(remaining.iter().all(|value| *value <= 1.0));
        Map2RewardState {
            remaining,
            tower_potential: self.tower_potential as f32,
            lane_potential: self.lane_potential as f32,
            lane_observed: self.lane_observed,
            fountain_wait_ticks: self.fountain_wait_ticks,
            fountain_wait_refundable_cost: self.fountain_wait_refundable_cost as f32,
            stagnation_ticks: self.stagnation_ticks,
            activity_ticks_left: self.activity_ticks_left,
            stagnation_base_charged: self.stagnation_base_charged,
            completed_tick: self.current.as_ref().map(|current| current.tick),
        }
    }

    fn ensure_live(&self) -> Result<(), Map2RewardError> {
        if self.ended {
            return invalid("episode already ended");
        }
        Ok(())
    }

    fn ensure_complete(&self) -> Result<(), Map2RewardError> {
        self.ensure_live()?;
        if self.pending.is_some() {
            return invalid("interval has pending Events");
        }
        if self.current.is_none() {
            return invalid("interval has no complete Snapshot/Events pair");
        }
        Ok(())
    }

    fn charge(&mut self, channel: usize, amount: u64) -> f64 {
        assert!(channel < MAP2_REWARD_CHANNELS);
        let before = self.counts[channel];
        self.counts[channel] = before
            .checked_add(amount)
            .expect("validated tick and amount bounds");
        let base = SCALES[channel] + before as f64;
        // Computing the increment directly preserves small costs near budget saturation.
        let emitted =
            BUDGETS[channel] * SCALES[channel] * amount as f64 / (base * (base + amount as f64));
        assert!(emitted >= 0.0);
        assert!(emitted <= BUDGETS[channel]);
        emitted
    }

    fn observe_resources(&mut self, pending: &SnapshotFacts) {
        let Some(previous) = &self.current else {
            return;
        };
        let xp = [
            pending.xp[0] - previous.xp[0],
            pending.xp[1] - previous.xp[1],
        ];
        let (mana, unobserved) = match (previous.mana, pending.mana) {
            (Some(before), Some(after))
                if before.id == after.id && before.maximum == after.maximum =>
            {
                ((before.mana - after.mana).max(0) as u64, false)
            }
            (None, None) => (0, false),
            _ => (0, true),
        };
        self.interval.observations.own_xp_gained += u64::from(xp[0]);
        self.interval.observations.enemy_xp_gained += u64::from(xp[1]);
        self.interval.experience +=
            self.charge(2, u64::from(xp[0])) - self.charge(3, u64::from(xp[1]));
        self.interval.observations.mana_spent += mana;
        self.interval.observations.mana_unobserved_ticks += u64::from(unobserved);
        self.interval.mana_spent -= self.charge(8, mana);
    }
}

impl Map2RewardBreakdown {
    fn retotal(&mut self) {
        self.total = self.gold
            + self.experience
            + self.hero_damage
            + self.hero_damage_taken
            + self.creep_damage_taken
            + self.other_damage_taken
            + self.mana_spent
            + self.tower_health
            + self.lane_pressure
            + self.pregame_movement
            + self.fountain_wait
            + self.fountain_wait_refund
            + self.stagnation_base
            + self.stagnation_ticks_cost
            + self.terminal;
        assert!(self.total.is_finite());
        assert!(self.ticks <= MAX_TICK);
    }
}

fn invalid<T>(message: &'static str) -> Result<T, Map2RewardError> {
    Err(Map2RewardError::Invalid(message))
}

fn limit(field: &'static str, actual: usize, maximum: usize) -> Result<(), Map2RewardError> {
    if actual > maximum {
        return Err(Map2RewardError::Limit {
            field,
            actual,
            maximum,
        });
    }
    Ok(())
}

const fn event_budget_total() -> f64 {
    let mut total = 0.0;
    let mut index = 0;
    while index < MAP2_REWARD_CHANNELS {
        assert!(BUDGETS[index] > 0.0);
        assert!(SCALES[index] > 0.0);
        total += BUDGETS[index];
        index += 1;
    }
    total
}

const fn schema_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut index = 0;
    while index < bytes.len() {
        hash ^= bytes[index] as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        index += 1;
    }
    hash
}
