use bota_proto::{
    AbilityId, AbilitySlot, Attribute, EffectId, EntityId, Fixed, ItemId, ItemView, Order, Target,
    Team, UnitKind, UnitView, Vec2,
};

use crate::teacher_economy::{self, EconomyObservation, attack_damage_against, holds_item};
use crate::{
    ActionError, ActionSpace, ActionTarget, ControlledUnit, EntityIndex, EntityRelation,
    IssuedOrder, ItemReadiness, OrderPersistence, PointIndex, PointSource, StateTracker,
    StructuredAction, TOWN_PORTAL_SCROLL, TacticalFeatures, TacticalPolicy,
};

const SHADOWRAZES: [(AbilityId, i32); 3] = [
    (AbilityId(13), 200),
    (AbilityId(14), 450),
    (AbilityId(15), 700),
];
const SHADOWRAZE_DAMAGE: [i32; 4] = [90, 160, 230, 300];
const REQUIEM: AbilityId = AbilityId(16);
const NECROMASTERY: AbilityId = AbilityId(17);
const PRESENCE: AbilityId = AbilityId(18);
const SOUL_EFFECT: EffectId = EffectId(11);
const REQUIEM_DAMAGE_PER_SOUL: [i32; 3] = [8, 11, 14];
const COURIER_BURST: AbilityId = AbilityId(8);
const COURIER_TAKE_STASH: AbilityId = AbilityId(10);
const COURIER_DELIVER: AbilityId = AbilityId(11);
const COURIER_SHIELD: AbilityId = AbilityId(12);
const BUILD_PLAN: [ItemId; 6] = teacher_economy::ECONOMY_PLAN;
const ATTACK_POINT_TICKS: u32 = 15;
const ATTACK_PROJECTILE_UNITS_PER_TICK: i32 = 40;
const TURN_RATE_BRADS: u32 = 5_795;
const ATTACK_ANGLE_BRADS: u16 = 2_094;
const ATTACK_RANGE_LEEWAY: i32 = 100;
const SHADOWRAZE_RADIUS: i32 = 250;
const REQUIEM_RADIUS: i32 = 900;
const ARMOR_SCALE: i64 = 6;
const TELEPORT_CHANNEL_TICKS: u32 = 90;
const COURIER_ERRAND_LIMIT_TICKS: u32 = 1_800;
const ORDER_NOTE_LIMIT: usize = 4;
const FOUNTAIN_RECOVERY_RADIUS: i32 = 1_200;
const FOUNTAIN_RECOVERY_PERCENT: i32 = 95;
const RETREAT_HEALTH_PERCENT: i32 = 40;
const BACKOFF_DISTANCE: i32 = 200;
const BACKOFF_THREAT_RANGE: i32 = 800;
const COMBAT_PLAN_TICKS: u32 = 90;
const AIM_PLAN_TICKS: u32 = 18;
const NAVIGATION_STALL_TICKS: u32 = 18;
const AGGRO_COOLDOWN_TICKS: u32 = 90;
const AGGRO_HOLD_TICKS: u32 = 70;
const AGGRO_RANGE: i32 = 500;
const FINISH_LIMIT_TICKS: u32 = 60;
const FINISH_ITEM_SLOTS: usize = 6;
const DECISION_TICKS: u32 = 3;
const COLLISION_MARGIN: i32 = DECISION_TICKS as i32 * 4;

const _: () = assert!(AGGRO_HOLD_TICKS < AGGRO_COOLDOWN_TICKS);
const _: () = assert!(AIM_PLAN_TICKS <= COMBAT_PLAN_TICKS);
const _: () = assert!(FINISH_LIMIT_TICKS <= COMBAT_PLAN_TICKS);
const _: () = assert!(FINISH_LIMIT_TICKS < 300);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CombatPurpose {
    Retreat,
    Finish { target: EntityId, until: u32 },
    Backoff(EntityId),
    Aim(EntityId),
    AggroClick { anchor: Vec2 },
    AggroPull,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CombatPlan {
    purpose: CombatPurpose,
    issued: IssuedOrder,
    origin: Vec2,
    tick: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct CombatMemory {
    hero: Option<EntityId>,
    active_items: [Option<ItemId>; 6],
    body: Option<OrderNote>,
    plan: Option<CombatPlan>,
    last_aggro: Option<u32>,
    finish_attempted: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CombatChoice {
    action: StructuredAction,
    purpose: Option<CombatPurpose>,
}

impl CombatChoice {
    const fn plain(action: StructuredAction) -> Self {
        Self {
            action,
            purpose: None,
        }
    }

    const fn planned(action: StructuredAction, purpose: CombatPurpose) -> Self {
        Self {
            action,
            purpose: Some(purpose),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OrderNote {
    sequence: u32,
    issued: IssuedOrder,
    tick: u32,
}

/// Deterministic bounded Shadow Fiend rule policy used as a training teacher.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Teacher {
    hero_notes: [Option<OrderNote>; ORDER_NOTE_LIMIT],
    hero_note_cursor: usize,
    courier_notes: [Option<OrderNote>; ORDER_NOTE_LIMIT],
    courier_note_cursor: usize,
    bought_once: [bool; BUILD_PLAN.len()],
    buy_sequences: [Option<u32>; BUILD_PLAN.len()],
    economy_observation: EconomyObservation,
    combat: CombatMemory,
    combat_proposal: Option<CombatPlan>,
    combat_rollback: Option<(u32, CombatMemory)>,
}

impl Default for Teacher {
    fn default() -> Self {
        Self::new()
    }
}

impl Teacher {
    /// Creates a teacher with empty per-match order and purchase memory.
    pub const fn new() -> Self {
        Self {
            hero_notes: [None; ORDER_NOTE_LIMIT],
            hero_note_cursor: 0,
            courier_notes: [None; ORDER_NOTE_LIMIT],
            courier_note_cursor: 0,
            bought_once: [false; BUILD_PLAN.len()],
            buy_sequences: [None; BUILD_PLAN.len()],
            economy_observation: EconomyObservation::new(),
            combat: CombatMemory {
                hero: None,
                active_items: [None; 6],
                body: None,
                plan: None,
                last_aggro: None,
                finish_attempted: false,
            },
            combat_proposal: None,
            combat_rollback: None,
        }
    }

    /// Selects an action and returns the exact action space used to select it.
    pub fn decide(
        &mut self,
        tracker: &StateTracker,
        persistence: &OrderPersistence,
        readiness: &ItemReadiness,
    ) -> Result<(StructuredAction, ActionSpace), ActionError> {
        self.decide_with_policy(tracker, persistence, readiness, None)
    }

    /// Applies a trained, seat-visible tactical residual without replacing economy or safety.
    /// Call `note_sent` / `note_rejected` and maintain persistence exactly as for `decide`.
    pub fn decide_tactical(
        &mut self,
        tracker: &StateTracker,
        persistence: &OrderPersistence,
        readiness: &ItemReadiness,
        policy: &TacticalPolicy,
    ) -> Result<(StructuredAction, ActionSpace), ActionError> {
        self.decide_with_policy(tracker, persistence, readiness, Some(policy))
    }

    fn decide_with_policy(
        &mut self,
        tracker: &StateTracker,
        persistence: &OrderPersistence,
        readiness: &ItemReadiness,
        policy: Option<&TacticalPolicy>,
    ) -> Result<(StructuredAction, ActionSpace), ActionError> {
        let space = ActionSpace::from_tracker_with_readiness(tracker, readiness)?;
        self.prepare_decision(tracker);
        let choice = self.priority_action(tracker, persistence, &space, policy);
        let selected = self.stage_choice(tracker, &space, choice)?;
        let decoded = space.decode(selected)?;
        let action = if decoded.is_some() && persistence.should_send(decoded).is_none() {
            if matches!(selected, StructuredAction::Stop { .. }) {
                self.combat.plan = None;
            }
            StructuredAction::Continue
        } else {
            selected
        };
        Ok((action, space))
    }

    /// Selects channel, sustain, bounded finishing, navigation cancellation, or retreat work.
    /// Stages bookkeeping for `note_sent`, just like `decide`.
    pub fn safety_action(
        &mut self,
        tracker: &StateTracker,
        space: &ActionSpace,
    ) -> Option<StructuredAction> {
        self.prepare_decision(tracker);
        let choice = self.mandatory_choice(tracker, space)?;
        Some(
            self.stage_choice(tracker, space, choice)
                .expect("mandatory Teacher action is legal"),
        )
    }

    /// Selects a high-confidence seat-visible action that deployment must not miss.
    pub fn deployment_action(
        &mut self,
        tracker: &StateTracker,
        space: &ActionSpace,
    ) -> Option<StructuredAction> {
        self.safety_action(tracker, space)
            .or_else(|| self.attack_structure(tracker, space))
    }

    fn prepare_decision(&mut self, tracker: &StateTracker) {
        self.economy_observation.observe(tracker);
        self.sync_purchases(tracker);
        self.sync_order_notes(tracker);
        self.sync_combat(tracker);
    }

    fn stage_choice(
        &mut self,
        tracker: &StateTracker,
        space: &ActionSpace,
        choice: CombatChoice,
    ) -> Result<StructuredAction, ActionError> {
        assert_eq!(tracker.current().map(|view| view.tick), Some(space.tick()));
        let choice = if space
            .decode(choice.action)?
            .is_some_and(|issued| issued.unit.is_none() && !tower_order_safe(tracker, issued.order))
        {
            CombatChoice::plain(StructuredAction::Stop {
                unit: ControlledUnit::Hero,
            })
        } else {
            choice
        };
        if !space.allows(choice.action) {
            return Err(ActionError::InvalidSchema("teacher selected masked action"));
        }
        let decoded = space.decode(choice.action)?;
        self.combat_proposal = choice.purpose.zip(decoded).and_then(|(purpose, issued)| {
            let previous = self.combat.plan.filter(|plan| plan.purpose == purpose);
            Some(CombatPlan {
                purpose,
                issued,
                origin: previous.map_or(tracker.own_hero()?.pos, |plan| plan.origin),
                tick: previous.map_or(space.tick(), |plan| plan.tick),
            })
        });
        if let Some(plan) = self.combat_proposal
            && matches!(plan.purpose, CombatPurpose::Finish { .. })
            && self
                .combat
                .body
                .is_some_and(|note| note.issued == plan.issued)
        {
            // Safety/deployment callers also suppress identical body orders without note_sent.
            self.accept_combat_plan(plan);
            self.combat_proposal = None;
        }
        Ok(choice.action)
    }

    fn mandatory_choice(
        &self,
        tracker: &StateTracker,
        space: &ActionSpace,
    ) -> Option<CombatChoice> {
        if self.protects_channel(tracker, space) {
            return Some(CombatChoice::plain(StructuredAction::Continue));
        }
        if let Some(action) = self
            .cancel_unsafe_order(tracker, space)
            .or_else(|| self.sustain(tracker, space))
        {
            return Some(CombatChoice::plain(action));
        }
        if let Some(choice) = self.finish_one_auto(tracker, space) {
            return Some(choice);
        }
        if let Some(action) = self.cancel_combat(tracker, space) {
            return Some(CombatChoice::plain(action));
        }
        let action = self.retreat(tracker, space)?;
        Some(if matches!(action, StructuredAction::MovePoint { .. }) {
            self.navigation_choice(action, CombatPurpose::Retreat)
        } else {
            CombatChoice::plain(action)
        })
    }

    /// Records a sent order and the snapshot tick of the space that decoded it.
    /// Item classification uses the latest decision, safety, or deployment snapshot.
    pub fn note_sent(&mut self, sequence: u32, issued: IssuedOrder, tick: u32) {
        self.note_combat_sent(sequence, issued, tick);
        if is_notable_order(issued.order) {
            let note = Some(OrderNote {
                sequence,
                issued,
                tick,
            });
            if issued.unit.is_some() {
                self.courier_notes[self.courier_note_cursor] = note;
                self.courier_note_cursor = (self.courier_note_cursor + 1) % ORDER_NOTE_LIMIT;
            } else {
                self.hero_notes[self.hero_note_cursor] = note;
                self.hero_note_cursor = (self.hero_note_cursor + 1) % ORDER_NOTE_LIMIT;
            }
        }
        if let Order::Buy { item } = issued.order
            && let Some(index) = BUILD_PLAN.iter().position(|planned| *planned == item)
            && item != TOWN_PORTAL_SCROLL
        {
            self.bought_once[index] = true;
            self.buy_sequences[index] = Some(sequence);
        }
    }

    /// Rolls back bounded local memory created by one rejected sequence.
    pub fn note_rejected(&mut self, sequence: u32) -> bool {
        let mut changed = false;
        if let Some((rejected, previous)) = self.combat_rollback
            && rejected == sequence
        {
            self.combat = previous;
            self.combat_rollback = None;
            changed = true;
        }
        changed |= reject_note(&mut self.hero_notes, sequence);
        changed |= reject_note(&mut self.courier_notes, sequence);
        for index in 0..BUILD_PLAN.len() {
            if self.buy_sequences[index] == Some(sequence) {
                self.bought_once[index] = false;
                self.buy_sequences[index] = None;
                changed = true;
            }
        }
        changed
    }

    fn priority_action(
        &self,
        tracker: &StateTracker,
        persistence: &OrderPersistence,
        space: &ActionSpace,
        policy: Option<&TacticalPolicy>,
    ) -> CombatChoice {
        if self.protects_channel(tracker, space) {
            return CombatChoice::plain(StructuredAction::Continue);
        }
        if let Some(action) = self.cancel_unsafe_order(tracker, space) {
            return CombatChoice::plain(action);
        }
        if let Some(action) = self
            .learn(tracker, space)
            .or_else(|| self.buy(tracker, space))
            .or_else(|| self.courier(tracker, space))
        {
            return CombatChoice::plain(action);
        }
        if let Some(choice) = self.mandatory_choice(tracker, space) {
            return choice;
        }
        if let Some(choice) = self.finish_aggro_click(tracker, space).or_else(|| {
            policy.and_then(|policy| self.tactical_action(tracker, persistence, space, policy))
        }) {
            return choice;
        }
        if let Some(action) = self
            .raze_enemy(tracker, space, UnitKind::Hero)
            .or_else(|| self.requiem(tracker, space))
            .or_else(|| self.raze_enemy(tracker, space, UnitKind::Tower))
        {
            return CombatChoice::plain(action);
        }
        if self.protects_active_unit_attack(tracker, persistence, space) {
            return CombatChoice::plain(StructuredAction::Continue);
        }
        if let Some(choice) = self
            .attack_last_hit(tracker, space)
            .map(CombatChoice::plain)
            .or_else(|| self.raze_last_hit(tracker, space))
            .or_else(|| self.deny(tracker, space).map(CombatChoice::plain))
            .or_else(|| self.harass(tracker, space).map(CombatChoice::plain))
            .or_else(|| {
                self.attack_structure(tracker, space)
                    .map(CombatChoice::plain)
            })
        {
            return choice;
        }
        if let Some(choice) = self
            .teacher_aim(tracker, space)
            .or_else(|| self.aggro_pull(tracker, space))
        {
            return choice;
        }
        if self.protects_useful_navigation(tracker, persistence) {
            return CombatChoice::plain(StructuredAction::Continue);
        }
        CombatChoice::plain(
            self.safe_objective(tracker, space)
                .or_else(|| self.hold_lane(tracker, space))
                .unwrap_or(StructuredAction::Continue),
        )
    }

    fn tactical_action(
        &self,
        tracker: &StateTracker,
        persistence: &OrderPersistence,
        space: &ActionSpace,
        policy: &TacticalPolicy,
    ) -> Option<CombatChoice> {
        let hero = tracker.own_hero().filter(|hero| hero.hp > 0)?;
        let (target, enemy) = combat_victim(tracker, space, hero)?;
        let last_hit = self.attack_last_hit(tracker, space);
        let features = tactical_features(
            tracker,
            space,
            hero,
            enemy,
            last_hit.is_some(),
            &self.combat,
        )?;
        let active_attack = self.protects_active_unit_attack(tracker, persistence, space);
        let fight = if active_attack {
            None
        } else {
            self.tactical_fight(tracker, space, hero, enemy, target, last_hit.is_some())
        };
        let farm = if active_attack {
            Some(CombatChoice::plain(StructuredAction::Continue))
        } else {
            last_hit
                .map(CombatChoice::plain)
                .or_else(|| self.raze_last_hit(tracker, space))
                .or_else(|| self.deny(tracker, space).map(CombatChoice::plain))
                .or_else(|| self.aggro_pull(tracker, space))
                .or_else(|| self.hold_lane(tracker, space).map(CombatChoice::plain))
                .or(Some(CombatChoice::plain(StructuredAction::Stop {
                    unit: ControlledUnit::Hero,
                })))
        };
        let recover = self.lane_backoff(tracker, space, hero, enemy);
        let alternatives = [None, fight, recover, farm];
        let mode = policy.choose(&features, alternatives.map(|action| action.is_some()));
        let action = alternatives[mode.index()];
        assert!(action.is_none_or(|choice| space.allows(choice.action)));
        action
    }

    fn tactical_fight(
        &self,
        tracker: &StateTracker,
        space: &ActionSpace,
        hero: &UnitView,
        enemy: &UnitView,
        target: EntityIndex,
        last_hit: bool,
    ) -> Option<CombatChoice> {
        let lethal = tactical_burst(hero, enemy, Some(space)) >= enemy.hp;
        let approach = if in_attack_reach(hero, enemy) {
            hero.pos
        } else {
            enemy.pos
        };
        if !tactical_chase_safe(tracker, hero, approach) || (!lethal && last_hit) {
            return None;
        }
        if let Some(action) = self.requiem(tracker, space) {
            return Some(CombatChoice::plain(action));
        }
        if let Some(choice) = self.aim_raze(tracker, space, hero, enemy, target) {
            return Some(choice);
        }
        if !in_attack_reach(hero, enemy) && (!lethal || hero.move_speed < enemy.move_speed) {
            return None;
        }
        let action = StructuredAction::AttackUnit {
            unit: ControlledUnit::Hero,
            target,
        };
        space.allows(action).then_some(CombatChoice::plain(action))
    }

    fn teacher_aim(&self, tracker: &StateTracker, space: &ActionSpace) -> Option<CombatChoice> {
        let hero = tracker.own_hero()?;
        let (target, enemy) = combat_victim(tracker, space, hero)?;
        if !tactical_chase_safe(tracker, hero, enemy.pos) {
            return None;
        }
        self.aim_raze(tracker, space, hero, enemy, target)
    }

    fn sync_combat(&mut self, tracker: &StateTracker) {
        let hero = tracker.own_hero().map(|hero| hero.id);
        if self.combat.hero.is_some() && self.combat.hero != hero {
            self.combat = CombatMemory::default();
            self.combat_rollback = None;
        }
        self.combat.hero = hero;
        if tracker
            .own_hero()
            .is_some_and(|hero| !ratio_at_most(hero.hp, hero.max_hp, RETREAT_HEALTH_PERCENT))
        {
            self.combat.finish_attempted = false;
        }
        self.combat.active_items = std::array::from_fn(|slot| {
            tracker
                .own_hero()?
                .items
                .get(slot)?
                .as_ref()
                .map(|item| item.id)
        });
        self.combat_proposal = None;
    }

    fn note_combat_sent(&mut self, sequence: u32, issued: IssuedOrder, tick: u32) {
        if issued.unit.is_some()
            || matches!(
                issued.order,
                Order::Learn { .. } | Order::Buy { .. } | Order::Sell { .. } | Order::Swap { .. }
            )
        {
            return;
        }
        self.combat_rollback = Some((sequence, self.combat));
        let preserving = self.preserves_body(issued.order);
        let proposal = self
            .combat_proposal
            .take()
            .filter(|plan| plan.issued == issued);
        if let Some(plan) = proposal {
            self.accept_combat_plan(plan);
        } else if !preserving
            || matches!(
                issued.order,
                Order::Cast {
                    target: Target::None,
                    ..
                }
            ) && self
                .combat
                .plan
                .is_some_and(|plan| matches!(plan.purpose, CombatPurpose::Aim(_)))
                && self.combat.body.is_some_and(|note| {
                    matches!(
                        note.issued.order,
                        Order::Move {
                            target: Target::None
                        }
                    )
                })
        {
            self.combat.plan = None;
        }
        match issued.order {
            Order::Move { .. } | Order::Attack { .. } => {
                self.combat.body = Some(OrderNote {
                    sequence,
                    issued,
                    tick,
                });
            }
            _ if preserving => {}
            _ => self.combat.body = None,
        }
    }

    fn accept_combat_plan(&mut self, plan: CombatPlan) {
        match plan.purpose {
            CombatPurpose::AggroClick { .. } => self.combat.last_aggro = Some(plan.tick),
            CombatPurpose::Finish { until, .. } => {
                assert!(until > plan.tick);
                assert!(until - plan.tick <= FINISH_LIMIT_TICKS);
                self.combat.finish_attempted = true;
            }
            _ => {}
        }
        self.combat.plan = Some(plan);
    }

    fn cancel_unsafe_order(
        &self,
        tracker: &StateTracker,
        space: &ActionSpace,
    ) -> Option<StructuredAction> {
        let note = self.combat.body?;
        if tower_order_safe(tracker, note.issued.order) {
            return None;
        }
        let stop = StructuredAction::Stop {
            unit: ControlledUnit::Hero,
        };
        space.allows(stop).then_some(stop)
    }

    fn finish_one_auto(&self, tracker: &StateTracker, space: &ActionSpace) -> Option<CombatChoice> {
        let hero = tracker.own_hero()?;
        if let Some(plan) = self.combat.plan
            && let CombatPurpose::Finish { target, until } = plan.purpose
        {
            let remaining = until.saturating_sub(space.tick());
            let viable = space.entity_index(target).is_some_and(|index| {
                space.allows(StructuredAction::AttackUnit {
                    unit: ControlledUnit::Hero,
                    target: index,
                }) && finish_risk_acceptable(
                    tracker,
                    hero,
                    space.entity_candidates()[index.0].unit(),
                    remaining,
                )
            });
            let action = if remaining > 0 && viable {
                StructuredAction::Continue
            } else {
                StructuredAction::Stop {
                    unit: ControlledUnit::Hero,
                }
            };
            return space.allows(action).then_some(CombatChoice::plain(action));
        }
        if !ratio_at_most(hero.hp, hero.max_hp, RETREAT_HEALTH_PERCENT)
            || self.combat.finish_attempted
            || own_fountain(tracker).is_some_and(|home| {
                hero.pos
                    .within(home, Fixed::from_int(FOUNTAIN_RECOVERY_RADIUS))
            })
        {
            return None;
        }
        space
            .entity_candidates()
            .iter()
            .enumerate()
            .find_map(|(index, candidate)| {
                let enemy = candidate.unit();
                if candidate.relation != EntityRelation::Enemy
                    || enemy.kind != UnitKind::Hero
                    || self.combat.body.is_some_and(|note| {
                        matches!(note.issued.order,
                    Order::Attack { target: Target::Unit(other) } if other != enemy.id)
                    })
                {
                    return None;
                }
                let ticks = finish_estimated_ticks(hero, enemy);
                let until = space.tick().checked_add(ticks)?;
                let action = StructuredAction::AttackUnit {
                    unit: ControlledUnit::Hero,
                    target: EntityIndex(index),
                };
                (space.allows(action) && finish_risk_acceptable(tracker, hero, enemy, ticks))
                    .then_some(CombatChoice::planned(
                        action,
                        CombatPurpose::Finish {
                            target: enemy.id,
                            until,
                        },
                    ))
            })
    }

    fn preserves_body(&self, order: Order) -> bool {
        match order {
            Order::Cast {
                target: Target::None,
                ..
            }
            | Order::Use {
                target: Target::None,
                ..
            } => true,
            Order::Use { slot, target } => match self
                .combat
                .active_items
                .get(usize::from(slot.0))
                .copied()
                .flatten()
            {
                Some(ItemId(7)) => matches!(target, Target::Pos(_)),
                Some(ItemId(1) | ItemId(2)) => {
                    matches!(target, Target::Unit(hero) if Some(hero) == self.combat.hero)
                }
                _ => false,
            },
            _ => false,
        }
    }

    fn navigation_choice(&self, action: StructuredAction, purpose: CombatPurpose) -> CombatChoice {
        if self.combat.plan.is_some_and(|plan| plan.purpose == purpose) {
            CombatChoice::plain(StructuredAction::Continue)
        } else {
            CombatChoice::planned(action, purpose)
        }
    }

    fn cancel_combat(
        &self,
        tracker: &StateTracker,
        space: &ActionSpace,
    ) -> Option<StructuredAction> {
        let plan = self.combat.plan?;
        let hero = tracker.own_hero()?;
        let elapsed = space.tick().saturating_sub(plan.tick);
        let invalid = match plan.purpose {
            CombatPurpose::Finish { until, .. } => space.tick() >= until,
            CombatPurpose::Retreat => self.retreat(tracker, space).is_none(),
            CombatPurpose::Backoff(target) => !enemy_heroes(tracker)
                .any(|enemy| enemy.id == target && backoff_needed(hero, enemy)),
            CombatPurpose::Aim(target) => {
                elapsed >= AIM_PLAN_TICKS
                    || space.entity_index(target).is_none_or(|index| {
                        let target = space.entity_candidates()[index.0].unit();
                        target.hp <= 0 || !hero.pos.within(target.pos, Fixed::from_int(1_200))
                    })
            }
            CombatPurpose::AggroClick { .. } | CombatPurpose::AggroPull => self
                .combat
                .last_aggro
                .is_some_and(|tick| space.tick().saturating_sub(tick) >= AGGRO_HOLD_TICKS),
        };
        let moving = matches!(
            plan.issued.order,
            Order::Move {
                target: Target::Pos(_) | Target::Unit(_)
            }
        );
        let arrived = match plan.issued.order {
            Order::Move {
                target: Target::Pos(point),
            } => hero.pos.within(point, Fixed::from_int(40)),
            _ => false,
        };
        let stalled = moving
            && elapsed >= NAVIGATION_STALL_TICKS
            && hero.pos.within(plan.origin, Fixed::from_int(8));
        let stop = StructuredAction::Stop {
            unit: ControlledUnit::Hero,
        };
        (invalid || arrived || stalled || elapsed >= COMBAT_PLAN_TICKS)
            .then_some(stop)
            .filter(|action| space.allows(*action))
    }

    fn lane_backoff(
        &self,
        tracker: &StateTracker,
        space: &ActionSpace,
        hero: &UnitView,
        enemy: &UnitView,
    ) -> Option<CombatChoice> {
        if !backoff_needed(hero, enemy) {
            return None;
        }
        let away = Vec2 {
            x: Fixed {
                raw: hero
                    .pos
                    .x
                    .raw
                    .saturating_add(hero.pos.x.raw.saturating_sub(enemy.pos.x.raw)),
            },
            y: Fixed {
                raw: hero
                    .pos
                    .y
                    .raw
                    .saturating_add(hero.pos.y.raw.saturating_sub(enemy.pos.y.raw)),
            },
        };
        let point = short_lane_point(tracker, space, away, Some(enemy.pos))?;
        Some(self.navigation_choice(
            StructuredAction::MovePoint {
                unit: ControlledUnit::Hero,
                point,
            },
            CombatPurpose::Backoff(enemy.id),
        ))
    }

    fn aim_raze(
        &self,
        tracker: &StateTracker,
        space: &ActionSpace,
        hero: &UnitView,
        enemy: &UnitView,
        target: EntityIndex,
    ) -> Option<CombatChoice> {
        let wanted = facing_towards(hero.pos, predicted_position(tracker, enemy));
        let slot = best_hero_raze(tracker, space, hero, enemy, hero.facing.brads)
            .or_else(|| best_hero_raze(tracker, space, hero, enemy, wanted))?;
        self.aim_raze_slot(tracker, space, hero, enemy, target, slot)
    }

    fn aim_raze_slot(
        &self,
        tracker: &StateTracker,
        space: &ActionSpace,
        hero: &UnitView,
        enemy: &UnitView,
        target: EntityIndex,
        slot: usize,
    ) -> Option<CombatChoice> {
        let reach = raze_reach(hero.abilities.get(slot)?.id)?;
        assert!(space.allows(cast_none(slot)));
        if raze_contains(tracker, hero, enemy, hero.facing.brads, reach) {
            let action = if self.facing_stable(tracker) {
                cast_none(slot)
            } else {
                StructuredAction::Stop {
                    unit: ControlledUnit::Hero,
                }
            };
            return space.allows(action).then_some(
                if matches!(action, StructuredAction::Cast { .. }) {
                    CombatChoice::plain(action)
                } else {
                    CombatChoice::planned(action, CombatPurpose::Aim(enemy.id))
                },
            );
        }
        let action = StructuredAction::FollowUnit {
            unit: ControlledUnit::Hero,
            target,
        };
        space
            .allows(action)
            .then_some(CombatChoice::planned(action, CombatPurpose::Aim(enemy.id)))
    }

    fn facing_stable(&self, tracker: &StateTracker) -> bool {
        let Some(hero) = tracker.own_hero() else {
            return false;
        };
        self.combat.body.is_some_and(|note| {
            matches!(
                note.issued.order,
                Order::Move {
                    target: Target::None
                }
            ) && tracker.current().is_some_and(|view| view.tick > note.tick)
        }) && tracker
            .entity(hero.id)
            .and_then(|entity| entity.velocity)
            .is_none_or(|velocity| velocity.delta == Vec2::ZERO)
    }

    fn aggro_pull(&self, tracker: &StateTracker, space: &ActionSpace) -> Option<CombatChoice> {
        let hero = tracker.own_hero()?;
        if self
            .combat
            .last_aggro
            .is_some_and(|tick| space.tick().saturating_sub(tick) < AGGRO_COOLDOWN_TICKS)
        {
            return self
                .combat
                .plan
                .filter(|plan| matches!(plan.purpose, CombatPurpose::AggroPull))
                .map(|_| CombatChoice::plain(StructuredAction::Continue));
        }
        if ratio_at_most(hero.hp, hero.max_hp, RETREAT_HEALTH_PERCENT)
            || self.attack_last_hit(tracker, space).is_some()
            || enemy_tower_danger(tracker, hero.pos, hero.radius)
        {
            return None;
        }
        let anchor = allied_ranged_anchor(tracker, hero.pos)?;
        short_lane_point(tracker, space, anchor, None)?;
        let view = tracker.current()?;
        let eligible = view.units.iter().any(|creep| {
            creep.team != hero.team
                && creep.team != Team::Neutral
                && creep.hp > 0
                && matches!(creep.kind, UnitKind::CreepMelee | UnitKind::CreepFlagbearer)
                && hero.pos.within(creep.pos, Fixed::from_int(AGGRO_RANGE))
                && early_aggro_eligible(tracker, creep)
                && anchor.distance_squared(creep.pos) > hero.pos.distance_squared(creep.pos)
        });
        if !eligible {
            return None;
        }
        let (target, _) = combat_victim(tracker, space, hero)?;
        let action = StructuredAction::AttackUnit {
            unit: ControlledUnit::Hero,
            target,
        };
        space.allows(action).then_some(CombatChoice::planned(
            action,
            CombatPurpose::AggroClick { anchor },
        ))
    }

    fn finish_aggro_click(
        &self,
        tracker: &StateTracker,
        space: &ActionSpace,
    ) -> Option<CombatChoice> {
        let plan = self.combat.plan?;
        let CombatPurpose::AggroClick { anchor } = plan.purpose else {
            return None;
        };
        if space.tick() <= plan.tick {
            return Some(CombatChoice::plain(StructuredAction::Continue));
        }
        if let Some(action) = self
            .attack_last_hit(tracker, space)
            .or_else(|| self.deny(tracker, space))
        {
            return Some(CombatChoice::plain(action));
        }
        let action = short_lane_point(tracker, space, anchor, None).map_or(
            StructuredAction::Stop {
                unit: ControlledUnit::Hero,
            },
            |point| StructuredAction::MovePoint {
                unit: ControlledUnit::Hero,
                point,
            },
        );
        space
            .allows(action)
            .then_some(CombatChoice::planned(action, CombatPurpose::AggroPull))
    }

    fn sync_purchases(&mut self, tracker: &StateTracker) {
        for (index, item) in BUILD_PLAN.iter().copied().enumerate() {
            if item != TOWN_PORTAL_SCROLL && holds_item(tracker, item) {
                self.bought_once[index] = true;
                self.buy_sequences[index] = None;
            }
        }
    }

    fn sync_order_notes(&mut self, tracker: &StateTracker) {
        if tracker
            .own_hero()
            .is_some_and(|hero| hero.statuses.bits & bota_proto::StatusFlags::STUNNED != 0)
        {
            for note in &mut self.hero_notes {
                if note.is_some_and(|note| is_item_use_order(note.issued.order)) {
                    *note = None;
                }
            }
        }
        let courier_idle = tracker.own_courier().is_some_and(|courier| {
            let bag_empty = courier.items.iter().all(Option::is_none);
            let home = own_fountain(tracker).is_some_and(|home| courier.pos == home);
            bag_empty && home
        });
        if courier_idle {
            for note in &mut self.courier_notes {
                if note.is_some_and(|note| is_courier_errand_order(tracker, note.issued.order)) {
                    *note = None;
                }
            }
        }
    }

    fn protects_channel(&self, tracker: &StateTracker, space: &ActionSpace) -> bool {
        if tracker
            .own_hero()
            .is_some_and(|hero| hero.statuses.bits & bota_proto::StatusFlags::CHANNELLING != 0)
        {
            return true;
        }
        if let Some(note) = self.latest_note(None) {
            let elapsed = space.tick().saturating_sub(note.tick);
            if let Order::Use { slot, .. } = note.issued.order
                && elapsed <= TELEPORT_CHANNEL_TICKS
                && space
                    .controlled_item(ControlledUnit::Hero, slot)
                    .is_some_and(|item| item.id == TOWN_PORTAL_SCROLL)
            {
                return true;
            }
        }
        false
    }

    fn protects_active_unit_attack(
        &self,
        tracker: &StateTracker,
        persistence: &OrderPersistence,
        space: &ActionSpace,
    ) -> bool {
        let Some(note) = self.active_note(persistence, None) else {
            return false;
        };
        if let Order::Attack {
            target: Target::Unit(target),
        } = note.issued.order
        {
            // Attack speed scales the interval, not Shadow Fiend's attack point.
            let windup = ATTACK_POINT_TICKS;
            let maximum_turn = 32_768u32.div_ceil(TURN_RATE_BRADS);
            return space.tick().saturating_sub(note.tick) <= windup.saturating_add(maximum_turn)
                && self.continue_attack(tracker, space, target);
        }
        false
    }

    fn protects_useful_navigation(
        &self,
        tracker: &StateTracker,
        persistence: &OrderPersistence,
    ) -> bool {
        let Some(note) = self.active_note(persistence, None) else {
            return false;
        };
        match note.issued.order {
            Order::Move {
                target: Target::Pos(target),
            }
            | Order::Attack {
                target: Target::Pos(target),
            } => useful_walk(tracker, target),
            _ => false,
        }
    }

    fn continue_attack(
        &self,
        tracker: &StateTracker,
        space: &ActionSpace,
        target: EntityId,
    ) -> bool {
        let Some(index) = space.entity_index(target) else {
            return false;
        };
        let Some(hero) = tracker.own_hero() else {
            return false;
        };
        let unit = space.entity_candidates()[index.0].unit();
        unit.hp > 0 && in_attack_reach_with_leeway(hero, unit)
    }

    fn active_note(
        &self,
        persistence: &OrderPersistence,
        unit: Option<EntityId>,
    ) -> Option<OrderNote> {
        let (sequence, issued) = persistence.active_body_for(unit)?;
        self.notes(unit)
            .iter()
            .flatten()
            .find(|note| note.sequence == sequence && note.issued == issued)
            .copied()
    }

    fn latest_note(&self, unit: Option<EntityId>) -> Option<OrderNote> {
        self.notes(unit)
            .iter()
            .flatten()
            .filter(|note| note.issued.unit == unit)
            .max_by_key(|note| note.sequence)
            .copied()
    }

    const fn notes(&self, unit: Option<EntityId>) -> &[Option<OrderNote>; ORDER_NOTE_LIMIT] {
        if unit.is_some() {
            &self.courier_notes
        } else {
            &self.hero_notes
        }
    }

    fn learn(&self, tracker: &StateTracker, space: &ActionSpace) -> Option<StructuredAction> {
        let hero = tracker.own_hero()?;
        let mask = space.learn_slot_mask();
        for wanted in [REQUIEM, SHADOWRAZES[0].0, NECROMASTERY, PRESENCE] {
            for (index, ability) in hero.abilities.iter().enumerate() {
                let same_group = wanted == SHADOWRAZES[0].0
                    && SHADOWRAZES.iter().any(|(id, _)| *id == ability.id);
                if (ability.id == wanted || same_group) && mask.get(index) == Some(&true) {
                    return Some(StructuredAction::Learn {
                        slot: AbilitySlot(index as u8),
                    });
                }
            }
        }
        None
    }

    fn buy(&self, tracker: &StateTracker, space: &ActionSpace) -> Option<StructuredAction> {
        teacher_economy::select_purchase(
            tracker,
            space,
            &self.bought_once,
            &self.economy_observation,
        )
    }

    fn courier(&self, tracker: &StateTracker, space: &ActionSpace) -> Option<StructuredAction> {
        let courier = tracker.own_courier()?;
        let carrying = courier.items.iter().any(Option::is_some);
        let stash_waiting = tracker
            .own_player()?
            .stash
            .as_ref()?
            .iter()
            .any(Option::is_some);
        if courier_threatened(tracker, courier)
            && let Some(action) = courier_cast(space, courier, COURIER_SHIELD)
                .or_else(|| courier_cast(space, courier, COURIER_BURST))
        {
            return Some(action);
        }
        if self.courier_errand_active(tracker, space, courier.id) {
            return None;
        }
        if carrying {
            return courier_cast(space, courier, COURIER_DELIVER)
                .or_else(|| courier_cast(space, courier, COURIER_BURST));
        }
        if stash_waiting {
            return courier_cast(space, courier, COURIER_TAKE_STASH);
        }
        None
    }

    fn courier_errand_active(
        &self,
        tracker: &StateTracker,
        space: &ActionSpace,
        courier: EntityId,
    ) -> bool {
        let Some(note) = self.latest_note(Some(courier)) else {
            return false;
        };
        if space.tick().saturating_sub(note.tick) > COURIER_ERRAND_LIMIT_TICKS {
            return false;
        }
        let Order::Cast { slot, .. } = note.issued.order else {
            return false;
        };
        tracker.own_courier().is_some_and(|body| {
            body.id == courier
                && body
                    .abilities
                    .get(usize::from(slot.0))
                    .is_some_and(|ability| {
                        matches!(ability.id, COURIER_TAKE_STASH | COURIER_DELIVER)
                    })
        })
    }

    fn sustain(&self, tracker: &StateTracker, space: &ActionSpace) -> Option<StructuredAction> {
        let hero = tracker.own_hero()?;
        let emergency = ratio_at_most(hero.hp, hero.max_hp, RETREAT_HEALTH_PERCENT)
            || visible_pressure(tracker, hero) >= hero.hp.max(0);
        teacher_economy::select_sustain(tracker, space, emergency)
    }

    fn retreat(&self, tracker: &StateTracker, space: &ActionSpace) -> Option<StructuredAction> {
        let hero = tracker.own_hero()?;
        let fountain = own_fountain(tracker)?;
        // Fountain recovery is fast; do not let combat or navigation pull us out half full.
        if hero
            .pos
            .within(fountain, Fixed::from_int(FOUNTAIN_RECOVERY_RADIUS))
            && (ratio_below(hero.hp, hero.max_hp, FOUNTAIN_RECOVERY_PERCENT)
                || ratio_below(hero.mana, hero.max_mana, FOUNTAIN_RECOVERY_PERCENT))
        {
            let action = StructuredAction::Hold {
                unit: ControlledUnit::Hero,
            };
            return space.allows(action).then_some(action);
        }
        let critical = ratio_at_most(hero.hp, hero.max_hp, RETREAT_HEALTH_PERCENT)
            && !safe_kill_opportunity(tracker, space, hero);
        let lethal = visible_pressure(tracker, hero) >= hero.hp.max(0);
        let tower = unsafe_tower_without_wave(tracker, hero);
        if !critical && !lethal && !tower {
            return None;
        }
        let point = best_safe_point(
            space,
            tracker,
            space.move_point_mask(ControlledUnit::Hero),
            fountain,
            true,
        )?;
        Some(StructuredAction::MovePoint {
            unit: ControlledUnit::Hero,
            point,
        })
    }

    fn attack_last_hit(
        &self,
        tracker: &StateTracker,
        space: &ActionSpace,
    ) -> Option<StructuredAction> {
        let target = best_attack_creep(tracker, space, EntityRelation::Enemy, false)?;
        Some(StructuredAction::AttackUnit {
            unit: ControlledUnit::Hero,
            target,
        })
    }

    fn raze_last_hit(&self, tracker: &StateTracker, space: &ActionSpace) -> Option<CombatChoice> {
        let hero = tracker.own_hero()?;
        let (target, slot) = best_farm_raze(tracker, space, hero)?;
        let creep = space.entity_candidates()[target.0].unit();
        self.aim_raze_slot(tracker, space, hero, creep, target, slot)
    }

    fn deny(&self, tracker: &StateTracker, space: &ActionSpace) -> Option<StructuredAction> {
        let target = best_attack_creep(tracker, space, EntityRelation::Allied, true)?;
        Some(StructuredAction::AttackUnit {
            unit: ControlledUnit::Hero,
            target,
        })
    }

    fn raze_enemy(
        &self,
        tracker: &StateTracker,
        space: &ActionSpace,
        kind: UnitKind,
    ) -> Option<StructuredAction> {
        let hero = tracker.own_hero()?;
        let view = tracker.current()?;
        if kind == UnitKind::Hero {
            let (target, enemy) = combat_victim(tracker, space, hero)?;
            if best_hero_raze(tracker, space, hero, enemy, hero.facing.brads).is_some() {
                return self
                    .aim_raze(tracker, space, hero, enemy, target)
                    .map(|choice| choice.action);
            }
            return None;
        }
        for (slot, ability) in hero.abilities.iter().enumerate() {
            let Some(reach) = raze_reach(ability.id) else {
                continue;
            };
            let action = cast_none(slot);
            if !space.allows(action) {
                continue;
            }
            let center = raze_center(hero.pos, hero.facing.brads, reach);
            if view.units.iter().any(|enemy| {
                enemy.kind == kind
                    && enemy.team != tracker.team()
                    && enemy.team != Team::Neutral
                    && enemy.hp > 0
                    && magical_damage(raze_damage(ability.level), enemy.magic_resist) > 0
                    && center.within(enemy.pos, Fixed::from_int(SHADOWRAZE_RADIUS))
            }) {
                return if self.facing_stable(tracker) {
                    Some(action)
                } else {
                    let stop = StructuredAction::Stop {
                        unit: ControlledUnit::Hero,
                    };
                    space.allows(stop).then_some(stop)
                };
            }
        }
        None
    }

    fn requiem(&self, tracker: &StateTracker, space: &ActionSpace) -> Option<StructuredAction> {
        let hero = tracker.own_hero()?;
        let souls = hero
            .effects
            .iter()
            .find(|effect| effect.id == SOUL_EFFECT)
            .and_then(|effect| effect.stacks)
            .unwrap_or(0);
        if souls < 8 || ratio_below(hero.hp, hero.max_hp, 45) {
            return None;
        }
        let (slot, ability) = hero
            .abilities
            .iter()
            .enumerate()
            .find(|(_, ability)| ability.id == REQUIEM)?;
        let action = cast_none(slot);
        if !space.allows(action) {
            return None;
        }
        let damage = requiem_damage(ability.level, souls);
        let valuable = enemy_heroes(tracker).any(|enemy| {
            let taken = magical_damage(damage, enemy.magic_resist);
            hero.pos.within(enemy.pos, Fixed::from_int(REQUIEM_RADIUS))
                && (taken >= enemy.hp / 4 || enemy.hp <= taken)
        });
        (valuable && !unsafe_tower_without_wave(tracker, hero)).then_some(action)
    }

    fn harass(&self, tracker: &StateTracker, space: &ActionSpace) -> Option<StructuredAction> {
        let hero = tracker.own_hero()?;
        if best_attack_creep(tracker, space, EntityRelation::Enemy, false).is_some()
            || enemy_tower_danger(tracker, hero.pos, hero.radius)
        {
            return None;
        }
        let mask = space.attack_entity_mask(ControlledUnit::Hero);
        space
            .entity_candidates()
            .iter()
            .enumerate()
            .filter(|(index, candidate)| {
                mask.get(*index) == Some(&true)
                    && candidate.relation == EntityRelation::Enemy
                    && candidate.kind == UnitKind::Hero
                    && in_attack_reach(hero, candidate.unit())
            })
            .min_by_key(|(_, candidate)| hero.pos.distance_squared(candidate.position))
            .map(|(index, _)| StructuredAction::AttackUnit {
                unit: ControlledUnit::Hero,
                target: EntityIndex(index),
            })
    }

    fn attack_structure(
        &self,
        tracker: &StateTracker,
        space: &ActionSpace,
    ) -> Option<StructuredAction> {
        let hero = tracker.own_hero()?;
        let mask = space.attack_entity_mask(ControlledUnit::Hero);
        space
            .entity_candidates()
            .iter()
            .enumerate()
            .filter(|(index, candidate)| {
                mask.get(*index) == Some(&true)
                    && candidate.relation == EntityRelation::Enemy
                    && matches!(
                        candidate.kind,
                        UnitKind::Tower | UnitKind::Ancient | UnitKind::Barracks
                    )
                    && in_attack_reach(hero, candidate.unit())
                    && allied_creep_near(tracker, candidate.position, 700)
            })
            .min_by_key(|(_, candidate)| hero.pos.distance_squared(candidate.position))
            .map(|(index, _)| StructuredAction::AttackUnit {
                unit: ControlledUnit::Hero,
                target: EntityIndex(index),
            })
    }

    fn hold_lane(&self, tracker: &StateTracker, space: &ActionSpace) -> Option<StructuredAction> {
        let hero = tracker.own_hero()?;
        let fountain = own_fountain(tracker)?;
        let wave = nearest_visible_creep(tracker, hero.pos)?;
        let wanted = point_along(wave.pos, fountain, Fixed::from_int(350));
        if hero.pos.within(wanted, Fixed::from_int(180)) {
            let hold = StructuredAction::Hold {
                unit: ControlledUnit::Hero,
            };
            return space.allows(hold).then_some(hold);
        }
        let point = best_safe_point(
            space,
            tracker,
            space.move_point_mask(ControlledUnit::Hero),
            wanted,
            false,
        )?;
        Some(StructuredAction::MovePoint {
            unit: ControlledUnit::Hero,
            point,
        })
    }

    fn safe_objective(
        &self,
        tracker: &StateTracker,
        space: &ActionSpace,
    ) -> Option<StructuredAction> {
        let hero = tracker.own_hero()?;
        let objective = enemy_objective(tracker, hero.pos)?;
        best_safe_point(
            space,
            tracker,
            space.attack_move_point_mask(ControlledUnit::Hero),
            objective,
            true,
        )
        .map(|point| StructuredAction::AttackMovePoint {
            unit: ControlledUnit::Hero,
            point,
        })
    }
}

#[allow(
    clippy::float_arithmetic,
    reason = "normalized seat-visible neural features"
)]
fn tactical_features(
    tracker: &StateTracker,
    space: &ActionSpace,
    hero: &UnitView,
    enemy: &UnitView,
    last_hit: bool,
    combat: &CombatMemory,
) -> Option<TacticalFeatures> {
    let ready = hero
        .abilities
        .iter()
        .enumerate()
        .filter(|(slot, ability)| {
            raze_reach(ability.id).is_some() && space.allows(cast_none(*slot))
        })
        .count();
    let distance =
        isqrt(hero.pos.distance_squared(enemy.pos) as u64) as f32 / Fixed::ONE.raw as f32 / 1_200.0;
    TacticalFeatures::from_values([
        tactical_ratio(hero.hp, hero.max_hp),
        tactical_ratio(enemy.hp, enemy.max_hp),
        tactical_ratio(hero.mana, hero.max_mana),
        tactical_ratio(enemy.mana, enemy.max_mana),
        tactical_ratio(
            physical_damage(hero.attack_damage.max(0), enemy.armor),
            enemy.hp,
        ),
        tactical_ratio(
            physical_damage(enemy.attack_damage.max(0), hero.armor),
            hero.hp,
        ),
        distance.clamp(0.0, 1.0),
        ready.min(3) as f32 / 3.0,
        tactical_ratio(tactical_burst(hero, enemy, Some(space)), enemy.hp),
        tactical_ratio(tactical_burst(enemy, hero, None), hero.hp),
        f32::from(enemy_tower_danger(tracker, hero.pos, hero.radius)),
        f32::from(enemy_tower_danger(tracker, enemy.pos, hero.radius)),
        tactical_ratio(visible_pressure(tracker, hero), hero.hp),
        f32::from(allied_creep_near(tracker, hero.pos, 750)),
        f32::from(last_hit),
        f32::from(in_attack_reach(hero, enemy)),
        f32::from(facing_gap(
            hero.facing.brads,
            facing_towards(hero.pos, enemy.pos),
        )) / 32_768.0,
        tactical_ratio(
            raze_hit_margin(tracker, space, hero, enemy),
            SHADOWRAZE_RADIUS,
        ),
        tactical_distance(hero.pos, predicted_position(tracker, enemy), 1_200),
        tactical_distance(
            hero.pos,
            allied_ranged_anchor(tracker, hero.pos).unwrap_or(enemy.pos),
            1_200,
        ),
        nearest_melee_distance(tracker, hero),
        combat.last_aggro.map_or(0.0, |tick| {
            AGGRO_COOLDOWN_TICKS.saturating_sub(space.tick().saturating_sub(tick)) as f32
                / AGGRO_COOLDOWN_TICKS as f32
        }),
        combat.plan.map_or(0.0, |plan| {
            combat_plan_remaining(plan, combat.last_aggro, space.tick()) as f32
                / COMBAT_PLAN_TICKS as f32
        }),
        combat.plan.map_or(0.0, |plan| {
            tactical_distance(hero.pos, plan.origin, BACKOFF_DISTANCE)
        }),
    ])
    .ok()
}

#[allow(
    clippy::float_arithmetic,
    reason = "bounded normalization with a nonzero denominator"
)]
fn tactical_ratio(value: i32, maximum: i32) -> f32 {
    let ratio = value.max(0) as f32 / maximum.max(1) as f32;
    assert!(ratio.is_finite());
    ratio.clamp(0.0, 1.0)
}

fn tactical_chase_safe(tracker: &StateTracker, hero: &UnitView, target: Vec2) -> bool {
    tower_corridor_safe(tracker, hero, target, false)
}

fn tower_order_safe(tracker: &StateTracker, order: Order) -> bool {
    let Some(hero) = tracker.own_hero() else {
        return true;
    };
    match order {
        Order::Move {
            target: Target::Pos(point),
        }
        | Order::Attack {
            target: Target::Pos(point),
        } => tower_corridor_safe(tracker, hero, point, true),
        Order::Move {
            target: Target::Unit(target),
        }
        | Order::Attack {
            target: Target::Unit(target),
        } => {
            let Some(target) = tracker
                .entity(target)
                .filter(|target| target.visible && target.unit.hp > 0)
            else {
                return false;
            };
            if matches!(order, Order::Attack { .. }) && in_attack_reach(hero, &target.unit) {
                return target.unit.kind != UnitKind::Hero
                    || tactical_chase_safe(tracker, hero, hero.pos);
            }
            tactical_chase_safe(tracker, hero, target.unit.pos)
        }
        _ => true,
    }
}

fn movement_guard(unit: &UnitView) -> Fixed {
    Fixed {
        raw: (unit.move_speed.raw.max(0) / 30)
            .saturating_mul(DECISION_TICKS as i32)
            .saturating_add(Fixed::from_int(COLLISION_MARGIN + 1).raw),
    }
}

fn tower_corridor_safe(
    tracker: &StateTracker,
    hero: &UnitView,
    target: Vec2,
    navigation: bool,
) -> bool {
    let horizontal = i128::from(target.x.raw) - i128::from(hero.pos.x.raw);
    let vertical = i128::from(target.y.raw) - i128::from(hero.pos.y.raw);
    let squared = horizontal * horizontal + vertical * vertical;
    // Wave-supported structure work remains possible, but never excuses hero pursuit.
    let pushing = navigation
        && allied_creep_near(tracker, target, 750)
        && !enemy_heroes(tracker).any(|enemy| {
            hero.pos.within(enemy.pos, Fixed::from_int(1_200))
                || target.within(enemy.pos, Fixed::from_int(1_200))
        });
    tracker.current().is_some_and(|view| {
        view.units
            .iter()
            .filter(|unit| {
                unit.kind == UnitKind::Tower && unit.team != tracker.team() && unit.hp > 0
            })
            .all(|tower| {
                let radius =
                    Fixed::from_int(700) + hero.radius + tower.radius + movement_guard(hero);
                let outward = (i128::from(hero.pos.x.raw) - i128::from(tower.pos.x.raw))
                    * horizontal
                    + (i128::from(hero.pos.y.raw) - i128::from(tower.pos.y.raw)) * vertical;
                if navigation && hero.pos.within(tower.pos, radius) && squared > 0 && outward >= 0 {
                    return true;
                }
                if pushing && target.within(tower.pos, radius) {
                    return true;
                }
                // Endpoints alone miss a chase corridor that cuts through tower range.
                let projection = ((i128::from(tower.pos.x.raw) - i128::from(hero.pos.x.raw))
                    * horizontal
                    + (i128::from(tower.pos.y.raw) - i128::from(hero.pos.y.raw)) * vertical)
                    .clamp(0, squared);
                let closest = Vec2 {
                    x: Fixed {
                        raw: (i128::from(hero.pos.x.raw) + horizontal * projection / squared.max(1))
                            as i32,
                    },
                    y: Fixed {
                        raw: (i128::from(hero.pos.y.raw) + vertical * projection / squared.max(1))
                            as i32,
                    },
                };
                !closest.within(tower.pos, radius)
            })
    })
}

fn tactical_burst(source: &UnitView, target: &UnitView, space: Option<&ActionSpace>) -> i32 {
    let disabled = source.statuses.bits
        & (bota_proto::StatusFlags::STUNNED | bota_proto::StatusFlags::SILENCED)
        != 0;
    let mut mana = source.mana;
    let raze = source
        .abilities
        .iter()
        .enumerate()
        .filter(|(slot, ability)| {
            !disabled
                && raze_reach(ability.id).is_some()
                && ability.level > 0
                && ability.cooldown_left == 0
                && ability.mana_cost <= source.mana
                && space.is_none_or(|space| space.allows(cast_none(*slot)))
        })
        .fold(0i32, |damage, (_, ability)| {
            if mana < ability.mana_cost {
                return damage;
            }
            mana -= ability.mana_cost;
            damage.saturating_add(magical_damage(
                raze_damage(ability.level),
                target.magic_resist,
            ))
        });
    physical_damage(source.attack_damage.max(0), target.armor)
        .saturating_mul(2)
        .saturating_add(raze)
}

fn combat_plan_remaining(plan: CombatPlan, last_aggro: Option<u32>, tick: u32) -> u32 {
    let (start, limit) = match plan.purpose {
        CombatPurpose::Finish { until, .. } => (plan.tick, until.saturating_sub(plan.tick)),
        CombatPurpose::Aim(_) => (plan.tick, AIM_PLAN_TICKS),
        CombatPurpose::AggroClick { .. } | CombatPurpose::AggroPull => {
            (last_aggro.unwrap_or(plan.tick), AGGRO_HOLD_TICKS)
        }
        CombatPurpose::Backoff(_) | CombatPurpose::Retreat => (plan.tick, COMBAT_PLAN_TICKS),
    };
    assert!(limit <= COMBAT_PLAN_TICKS);
    let remaining = limit.saturating_sub(tick.saturating_sub(start));
    assert!(remaining <= limit);
    remaining
}

fn combat_victim<'a>(
    tracker: &StateTracker,
    space: &'a ActionSpace,
    hero: &UnitView,
) -> Option<(EntityIndex, &'a UnitView)> {
    space
        .entity_candidates()
        .iter()
        .enumerate()
        .filter(|(_, candidate)| {
            candidate.relation == EntityRelation::Enemy
                && candidate.kind == UnitKind::Hero
                && candidate.unit().hp > 0
                && hero.pos.within(candidate.position, Fixed::from_int(1_200))
        })
        .min_by_key(|(_, candidate)| {
            let enemy = candidate.unit();
            let killable = tactical_burst(hero, enemy, Some(space)) >= enemy.hp
                && tactical_chase_safe(tracker, hero, enemy.pos);
            (
                !killable,
                hero.pos.distance_squared(enemy.pos),
                enemy.hp,
                enemy.id,
            )
        })
        .map(|(index, candidate)| (EntityIndex(index), candidate.unit()))
}

fn backoff_needed(hero: &UnitView, enemy: &UnitView) -> bool {
    enemy.hp > 0
        && hero
            .pos
            .within(enemy.pos, Fixed::from_int(BACKOFF_THREAT_RANGE))
        && (ratio_at_most(hero.hp, hero.max_hp, 80) || ratio_at_most(hero.mana, hero.max_mana, 25))
}

fn short_lane_point(
    tracker: &StateTracker,
    space: &ActionSpace,
    wanted: Vec2,
    away: Option<Vec2>,
) -> Option<PointIndex> {
    let hero = tracker.own_hero()?;
    let current = hero.pos.distance_squared(wanted);
    let anchor = allied_ranged_anchor(tracker, hero.pos);
    space
        .point_candidates()
        .iter()
        .enumerate()
        .filter(|(index, point)| {
            space.move_point_mask(ControlledUnit::Hero).get(*index) == Some(&true)
                && matches!(
                    point.source,
                    PointSource::Tactical {
                        radius: BACKOFF_DISTANCE,
                        ..
                    }
                )
                && point.position.distance_squared(wanted) < current
                && away.is_none_or(|enemy| {
                    point.position.distance_squared(enemy) > hero.pos.distance_squared(enemy)
                })
                && tactical_chase_safe(tracker, hero, point.position)
                && anchor.is_none_or(|anchor| {
                    point.position.distance_squared(anchor)
                        <= hero
                            .pos
                            .distance_squared(anchor)
                            .max(i64::from(Fixed::from_int(750).raw).pow(2))
                })
        })
        .min_by_key(|(_, point)| point.position.distance_squared(wanted))
        .map(|(index, _)| PointIndex(index))
}

fn allied_ranged_anchor(tracker: &StateTracker, position: Vec2) -> Option<Vec2> {
    tracker
        .current()?
        .units
        .iter()
        .filter(|unit| {
            unit.team == tracker.team()
                && unit.kind == UnitKind::CreepRanged
                && unit.hp > 0
                && position.within(unit.pos, Fixed::from_int(800))
        })
        .min_by_key(|unit| (position.distance_squared(unit.pos), unit.id))
        .map(|unit| unit.pos)
}

fn early_aggro_eligible(tracker: &StateTracker, creep: &UnitView) -> bool {
    let Some(view) = tracker.current() else {
        return false;
    };
    if view.tick >= tracker.metadata().pregame_ticks.saturating_add(9_000) {
        return true;
    }
    view.units.iter().any(|unit| {
        unit.hp > 0
            && (((is_lane_creep(unit.kind) && unit.team != creep.team)
                || unit.team == Team::Neutral)
                && creep.pos.within(unit.pos, Fixed::from_int(AGGRO_RANGE)))
    })
}

pub(crate) fn predicted_position(tracker: &StateTracker, unit: &UnitView) -> Vec2 {
    let Some(velocity) = tracker.entity(unit.id).and_then(|entity| entity.velocity) else {
        return unit.pos;
    };
    assert!(velocity.elapsed_ticks > 0);
    let divisor = i64::from(velocity.elapsed_ticks);
    let delta = Vec2 {
        x: Fixed {
            raw: (i64::from(velocity.delta.x.raw) / divisor) as i32,
        },
        y: Fixed {
            raw: (i64::from(velocity.delta.y.raw) / divisor) as i32,
        },
    };
    let distance = isqrt(delta.distance_squared(Vec2::ZERO) as u64).min(i32::MAX as u64) as i32;
    let step = Fixed {
        raw: distance.min(unit.move_speed.raw.max(0) / 30),
    };
    let maximum = (i64::from(tracker.metadata().terrain_cells)
        * i64::from(crate::TERRAIN_CELL_SIZE)
        * i64::from(Fixed::ONE.raw)
        - 1)
    .min(i64::from(i32::MAX)) as i32;
    assert!(maximum > 0);
    let target = Vec2 {
        x: Fixed {
            raw: unit.pos.x.raw.saturating_add(delta.x.raw).clamp(0, maximum),
        },
        y: Fixed {
            raw: unit.pos.y.raw.saturating_add(delta.y.raw).clamp(0, maximum),
        },
    };
    let available = isqrt(unit.pos.distance_squared(target) as u64).min(i32::MAX as u64) as i32;
    let predicted = point_along(
        unit.pos,
        target,
        Fixed {
            raw: step.raw.min(available),
        },
    );
    assert!(predicted.x.raw >= 0);
    assert!(predicted.x.raw <= maximum);
    assert!(predicted.y.raw >= 0);
    assert!(predicted.y.raw <= maximum);
    predicted
}

fn best_hero_raze(
    tracker: &StateTracker,
    space: &ActionSpace,
    hero: &UnitView,
    enemy: &UnitView,
    facing: u16,
) -> Option<usize> {
    let predicted = predicted_position(tracker, enemy);
    let uncertainty = enemy.move_speed.raw.max(0) / 30;
    let radius = Fixed {
        raw: Fixed::from_int(SHADOWRAZE_RADIUS)
            .raw
            .saturating_sub(uncertainty)
            .max(0),
    };
    hero.abilities
        .iter()
        .enumerate()
        .filter_map(|(slot, ability)| {
            let reach = raze_reach(ability.id)?;
            let center = raze_center(hero.pos, facing, reach);
            (space.allows(cast_none(slot))
                && magical_damage(raze_damage(ability.level), enemy.magic_resist) > 0
                && center.within(predicted, radius))
            .then_some((center.distance_squared(predicted), slot))
        })
        .min()
        .map(|(_, slot)| slot)
}

fn safe_kill_opportunity(tracker: &StateTracker, space: &ActionSpace, hero: &UnitView) -> bool {
    if enemy_tower_danger(tracker, hero.pos, hero.radius)
        || visible_pressure(tracker, hero).saturating_mul(2) >= hero.hp
    {
        return false;
    }
    enemy_heroes(tracker).any(|enemy| {
        if tactical_burst(enemy, hero, None) >= hero.hp {
            return false;
        }
        best_hero_raze(tracker, space, hero, enemy, hero.facing.brads).is_some_and(|slot| {
            magical_damage(raze_damage(hero.abilities[slot].level), enemy.magic_resist) >= enemy.hp
        })
    })
}

// This is a bounded trial estimate, not an observed attack phase. Half an interval
// budgets unknown cooldown; the fixed deadline and range abort prevent an open-ended chase.
fn finish_estimated_ticks(hero: &UnitView, enemy: &UnitView) -> u32 {
    let gap = u32::from(facing_gap(
        hero.facing.brads,
        facing_towards(hero.pos, enemy.pos),
    ));
    let turn = gap
        .saturating_sub(u32::from(ATTACK_ANGLE_BRADS))
        .div_ceil(TURN_RATE_BRADS);
    let distance = isqrt(hero.pos.distance_squared(enemy.pos) as u64);
    let travel =
        distance.div_ceil((Fixed::ONE.raw * ATTACK_PROJECTILE_UNITS_PER_TICK) as u64) as u32;
    hero.attack_interval
        .div_ceil(2)
        .max(turn)
        .max(1)
        .saturating_add(ATTACK_POINT_TICKS)
        .saturating_add(travel.saturating_sub(1))
        .saturating_add(DECISION_TICKS)
}

fn finish_risk_acceptable(
    tracker: &StateTracker,
    hero: &UnitView,
    enemy: &UnitView,
    ticks: u32,
) -> bool {
    if ticks == 0 || ticks > FINISH_LIMIT_TICKS || hero.hp <= 0 || enemy.hp <= 0 {
        return false;
    }
    assert!(ticks <= FINISH_LIMIT_TICKS);
    let disabled = bota_proto::StatusFlags::STUNNED
        | bota_proto::StatusFlags::DISARMED
        | bota_proto::StatusFlags::DOT
        | bota_proto::StatusFlags::CHANNELLING;
    let reach = hero.attack_range + hero.radius + enemy.radius - movement_guard(enemy);
    // Keep restoration separate from the wire-truncation and natural-regeneration margin.
    let health_margin = (2 + ticks.div_ceil(10) as i32)
        .saturating_add(finish_item_restoration(tracker, enemy, ticks));
    if hero.statuses.bits & disabled != 0
        || enemy
            .effects
            .iter()
            .any(|effect| matches!(effect.id, EffectId(1) | EffectId(3)))
        || !hero.pos.within(enemy.pos, reach.max(Fixed::ZERO))
        || !tactical_chase_safe(tracker, hero, hero.pos)
        || physical_damage(hero.attack_damage.max(0), enemy.armor)
            < enemy.hp.saturating_add(health_margin)
    {
        return false;
    }
    let Some(view) = tracker.current() else {
        return false;
    };
    // The wire omits projectile targets and launch damage. Nearby hostile flight is a veto,
    // not proof it targets us; newly launched enemy shots can also be absent for one tick.
    if view
        .projectiles
        .iter()
        .any(|shot| shot.team != hero.team && hero.pos.within(shot.pos, Fixed::from_int(1_200)))
    {
        return false;
    }
    let Some(spells) = enemy_heroes(tracker)
        .filter(|source| hero.pos.within(source.pos, Fixed::from_int(1_200)))
        .try_fold(0i32, |damage, source| {
            Some(damage.saturating_add(finish_raze_budget(tracker, source, hero, ticks)?))
        })
    else {
        return false;
    };
    finish_reply_estimate(view, hero, ticks)
        .saturating_add(spells)
        .saturating_add(1)
        < hero.hp
}

fn finish_raze_budget(
    tracker: &StateTracker,
    source: &UnitView,
    hero: &UnitView,
    ticks: u32,
) -> Option<i32> {
    assert!(ticks > 0);
    assert!(ticks <= FINISH_LIMIT_TICKS);
    let mana = finish_mana_budget(tracker, source, ticks);
    let mut casts = [(0i32, 0i32); SHADOWRAZES.len()];
    for ability in &source.abilities {
        if ability.passive
            || ability.level == 0
            || ability.cooldown_left > ticks
            || ability.mana_cost > mana
        {
            continue;
        }
        let (index, &(_, reach)) = SHADOWRAZES
            .iter()
            .enumerate()
            .find(|(_, (id, _))| *id == ability.id)?;
        if source.hero != Some(crate::SHADOW_FIEND) || ability.level > 4 {
            return None;
        }
        if possible_finish_raze(source, hero, reach, ticks) {
            casts[index] = (
                ability.mana_cost.max(0),
                magical_damage(raze_damage(ability.level), hero.magic_resist),
            );
        }
    }
    // All eight subsets fit in constant work. Slot-order greed can spend mana on a weaker raze.
    // Each raze's 300-tick cooldown exceeds the entire finish window, so it can cast only once.
    let mut maximum = 0;
    for subset in 0..(1 << SHADOWRAZES.len()) {
        let (mut cost, mut damage) = (0i32, 0i32);
        for (index, &(mana_cost, hit)) in casts.iter().enumerate() {
            if subset & (1 << index) != 0 {
                cost = cost.saturating_add(mana_cost);
                damage = damage.saturating_add(hit);
            }
        }
        if cost <= mana {
            maximum = maximum.max(damage);
        }
    }
    Some(maximum)
}

fn possible_finish_raze(source: &UnitView, hero: &UnitView, reach: i32, ticks: u32) -> bool {
    assert!(SHADOWRAZES.iter().any(|(_, distance)| *distance == reach));
    assert!(ticks <= FINISH_LIMIT_TICKS);
    let distance = isqrt(source.pos.distance_squared(hero.pos) as u64) as i64;
    // Facing is not a promise: allow the caster to turn/walk, plus both bodies' separation.
    let movement = (i64::from(source.move_speed.raw.max(0)) / 30
        + i64::from(Fixed::from_int(8).raw))
        * i64::from(ticks);
    let radius = i64::from(Fixed::from_int(SHADOWRAZE_RADIUS).raw) + movement + 1;
    (distance - i64::from(Fixed::from_int(reach).raw)).abs() <= radius
}

fn finish_mana_budget(tracker: &StateTracker, source: &UnitView, ticks: u32) -> i32 {
    assert!(ticks > 0);
    assert!(ticks <= FINISH_LIMIT_TICKS);
    // Fountain replenishment is outside this short lane estimate.
    if source.effects.iter().any(|effect| effect.id == EffectId(3)) {
        return i32::MAX;
    }
    // Audited SF regeneration is (0.25 + intelligence / 20 + Sage's Masks) mana/second.
    // Readiness within the window permits a whole-window upper estimate of regeneration.
    let masks = finish_eligible_items(source, ticks)
        .filter(|item| item.id == ItemId(26))
        .count() as u64;
    let treads = finish_eligible_items(source, ticks)
        .filter(|item| item.id == ItemId(29) && item.mode != Some(Attribute::Intelligence))
        .count() as u64;
    let whole = Fixed::ONE.raw as u64;
    let rate =
        (5 + 20 * masks + 10 * treads) * whole + source.attributes.intelligence.raw.max(0) as u64;
    let regenerated = (rate * u64::from(ticks)).div_ceil(600 * whole);
    source
        .mana
        .max(0)
        .saturating_add(1)
        .saturating_add(regenerated.min(i32::MAX as u64) as i32)
        // Ten additional intelligence can increase the mana pool by at most 120 per Treads.
        .saturating_add((120 * treads) as i32)
        .saturating_add(finish_item_restoration(tracker, source, ticks))
        .saturating_add(finish_clarity_mana(source, ticks))
}

fn finish_eligible_items(source: &UnitView, ticks: u32) -> impl Iterator<Item = &ItemView> {
    assert!(ticks > 0);
    assert!(ticks <= FINISH_LIMIT_TICKS);
    source
        .items
        .iter()
        .take(FINISH_ITEM_SLOTS)
        .flatten()
        .filter(move |item| item.cooldown_left <= ticks && item.mute_left <= ticks)
}

fn finish_item_restoration(tracker: &StateTracker, source: &UnitView, ticks: u32) -> i32 {
    let can_recharge = finish_can_recharge(tracker, source, ticks);
    // Without another visible affordable caster, use observed charges, not imagined full items.
    let restoration = finish_eligible_items(source, ticks).fold(0i32, |restored, item| {
        let maximum = match item.id {
            ItemId(35) => 10,
            ItemId(36) => 20,
            ItemId(id) if id > 41 => return i32::MAX,
            _ => return restored,
        };
        let charges = if can_recharge {
            maximum
        } else {
            item.charges.unwrap_or(0).min(maximum)
        };
        restored.saturating_add(15 * i32::from(charges))
    });
    assert!(restoration >= 0);
    restoration
}

fn finish_can_recharge(tracker: &StateTracker, source: &UnitView, ticks: u32) -> bool {
    assert!(ticks > 0);
    assert!(ticks <= FINISH_LIMIT_TICKS);
    let own = tracker.own_hero().map(|hero| hero.id);
    // The finishing hero is committed to an auto. Other visible allied casters may still cast;
    // without their future orders/cooldowns, item capacity is the finite replenishment ceiling.
    tracker.current().is_some_and(|view| {
        view.units.iter().any(|caster| {
            caster.kind == UnitKind::Hero
                && caster.team == tracker.team()
                && Some(caster.id) != own
                && caster.hp > 0
                && caster.pos.within(source.pos, Fixed::from_int(1_200))
                && caster.abilities.iter().any(|ability| {
                    !ability.passive
                        && ability.level > 0
                        && ability.cooldown_left <= ticks
                        && ability.mana_cost <= caster.mana
                })
        })
    })
}

fn finish_clarity_mana(source: &UnitView, ticks: u32) -> i32 {
    assert!(ticks > 0);
    assert!(ticks <= FINISH_LIMIT_TICKS);
    let active = source
        .effects
        .iter()
        .filter(|effect| effect.id == EffectId(2))
        .map(|effect| effect.ticks_left.unwrap_or(ticks).min(ticks))
        .max()
        .unwrap_or(0);
    let duration = if finish_eligible_items(source, ticks)
        .any(|item| item.id == ItemId(1) && item.charges.is_some_and(|charges| charges > 0))
    {
        ticks
    } else {
        active
    };
    // Clarity restores 150 over 750 ticks; refreshing it does not stack another regeneration.
    duration.div_ceil(5) as i32
}

// One reply may already be winding up, including beyond normal acquisition range.
// Killing its source does not cancel an already launched projectile.
fn finish_reply_estimate(view: &bota_proto::WorldView, hero: &UnitView, ticks: u32) -> i32 {
    assert!(ticks > 0);
    assert!(ticks <= FINISH_LIMIT_TICKS);
    let incoming = view
        .units
        .iter()
        .filter(|source| {
            source.team != hero.team
                && source.hp > 0
                && source.attack_damage > 0
                && hero.pos.within(
                    source.pos,
                    source.attack_range
                        + source.radius
                        + hero.radius
                        + movement_guard(source)
                        + Fixed::from_int(ATTACK_RANGE_LEEWAY),
                )
        })
        .fold(0i32, |damage, source| {
            let replies = 1 + ticks.saturating_sub(1) / source.attack_interval.max(1);
            damage.saturating_add(
                physical_damage(source.attack_damage, hero.armor).saturating_mul(replies as i32),
            )
        });
    assert!(incoming >= 0);
    incoming
}

fn raze_hit_margin(
    tracker: &StateTracker,
    space: &ActionSpace,
    hero: &UnitView,
    enemy: &UnitView,
) -> i32 {
    let Some(slot) = best_hero_raze(tracker, space, hero, enemy, hero.facing.brads) else {
        return 0;
    };
    let reach = raze_reach(hero.abilities[slot].id).expect("selected raze");
    let center = raze_center(hero.pos, hero.facing.brads, reach);
    let distance = isqrt(center.distance_squared(predicted_position(tracker, enemy)) as u64)
        / Fixed::ONE.raw as u64;
    SHADOWRAZE_RADIUS.saturating_sub(distance as i32)
}

#[allow(
    clippy::float_arithmetic,
    reason = "bounded normalized visible geometry"
)]
fn tactical_distance(source: Vec2, target: Vec2, maximum: i32) -> f32 {
    assert!(maximum > 0);
    let distance = isqrt(source.distance_squared(target) as u64) as f32 / Fixed::ONE.raw as f32;
    let normalized = (distance / maximum as f32).clamp(0.0, 1.0);
    assert!(normalized.is_finite());
    normalized
}

fn nearest_melee_distance(tracker: &StateTracker, hero: &UnitView) -> f32 {
    tracker
        .current()
        .and_then(|view| {
            view.units
                .iter()
                .filter(|unit| {
                    unit.hp > 0
                        && unit.team != hero.team
                        && unit.team != Team::Neutral
                        && matches!(unit.kind, UnitKind::CreepMelee | UnitKind::CreepFlagbearer)
                })
                .min_by_key(|unit| (hero.pos.distance_squared(unit.pos), unit.id))
        })
        .map_or(1.0, |unit| {
            tactical_distance(hero.pos, unit.pos, AGGRO_RANGE)
        })
}

fn courier_cast(
    space: &ActionSpace,
    courier: &UnitView,
    ability_id: AbilityId,
) -> Option<StructuredAction> {
    let slot = courier
        .abilities
        .iter()
        .position(|ability| ability.id == ability_id)?;
    let action = StructuredAction::Cast {
        unit: ControlledUnit::Courier,
        slot: AbilitySlot(slot as u8),
        target: ActionTarget::None,
    };
    space.allows(action).then_some(action)
}

fn useful_walk(tracker: &StateTracker, target: Vec2) -> bool {
    let Some(hero) = tracker.own_hero() else {
        return false;
    };
    if hero.pos.within(target, Fixed::from_int(100)) {
        return false;
    }
    let healthy = !ratio_at_most(hero.hp, hero.max_hp, 25);
    let pressure_safe = visible_pressure(tracker, hero) < hero.hp.max(0);
    healthy && pressure_safe && tower_corridor_safe(tracker, hero, target, true)
}

fn courier_threatened(tracker: &StateTracker, courier: &UnitView) -> bool {
    tracker.current().is_some_and(|view| {
        view.units.iter().any(|unit| {
            unit.team != tracker.team()
                && unit.team != Team::Neutral
                && unit.attack_damage > 0
                && courier
                    .pos
                    .within(unit.pos, unit.attack_range + unit.radius + courier.radius)
        })
    })
}

fn ratio_at_most(value: i32, maximum: i32, percent: i32) -> bool {
    maximum > 0 && i64::from(value) * 100 <= i64::from(maximum) * i64::from(percent)
}

fn ratio_below(value: i32, maximum: i32, percent: i32) -> bool {
    maximum <= 0 || i64::from(value) * 100 < i64::from(maximum) * i64::from(percent)
}

fn visible_pressure(tracker: &StateTracker, hero: &UnitView) -> i32 {
    let Some(view) = tracker.current() else {
        return 0;
    };
    view.units
        .iter()
        .filter(|unit| {
            unit.team != tracker.team()
                && unit.team != Team::Neutral
                && unit.attack_damage > 0
                && hero
                    .pos
                    .within(unit.pos, unit.attack_range + unit.radius + hero.radius)
        })
        .fold(0, |sum, unit| {
            sum.saturating_add(physical_damage(unit.attack_damage, hero.armor))
        })
}

fn physical_damage(amount: i32, armor: Fixed) -> i32 {
    let armor_raw = i64::from(armor.raw.max(0));
    let whole = i64::from(Fixed::ONE.raw);
    let denominator = 100 * whole + ARMOR_SCALE * armor_raw;
    (i64::from(amount) * 100 * whole / denominator) as i32
}

fn magical_damage(amount: i32, resistance: Fixed) -> i32 {
    let kept = i64::from(
        Fixed::ONE
            .raw
            .saturating_sub(resistance.raw)
            .clamp(0, Fixed::ONE.raw),
    );
    (i64::from(amount) * kept / i64::from(Fixed::ONE.raw)) as i32
}

fn best_attack_creep(
    tracker: &StateTracker,
    space: &ActionSpace,
    relation: EntityRelation,
    deny: bool,
) -> Option<EntityIndex> {
    let hero = tracker.own_hero()?;
    let mask = space.attack_entity_mask(ControlledUnit::Hero);
    space
        .entity_candidates()
        .iter()
        .enumerate()
        .filter_map(|(index, candidate)| {
            let unit = candidate.unit();
            let neutral = !deny
                && relation == EntityRelation::Enemy
                && candidate.kind == UnitKind::CreepNeutral
                && matches!(
                    candidate.relation,
                    EntityRelation::Enemy | EntityRelation::Neutral
                );
            if mask.get(index) != Some(&true)
                || !(neutral || candidate.relation == relation && is_lane_creep(candidate.kind))
                || !in_attack_reach(hero, unit)
                || (deny && unit.hp.saturating_mul(2) >= unit.max_hp)
            {
                return None;
            }
            let landing = attack_landing_ticks(hero, unit);
            let predicted = predicted_hp(tracker, unit, landing);
            let hit = physical_damage(attack_damage_against(hero, unit), unit.armor);
            (predicted > 0 && predicted <= hit).then_some((predicted, index))
        })
        .min()
        .map(|(_, index)| EntityIndex(index))
}

fn predicted_hp(tracker: &StateTracker, unit: &UnitView, ticks: u32) -> i32 {
    let Some(track) = tracker.entity(unit.id) else {
        return unit.hp;
    };
    let elapsed = track
        .last_seen_tick
        .saturating_sub(track.previous_seen_tick);
    if elapsed == 0 || track.hp_delta >= 0 {
        return unit.hp;
    }
    let change = track.hp_delta.saturating_mul(i64::from(ticks)) / i64::from(elapsed);
    i64::from(unit.hp)
        .saturating_add(change)
        .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

fn attack_landing_ticks(hero: &UnitView, target: &UnitView) -> u32 {
    let wanted = facing_towards(hero.pos, target.pos);
    let gap = u32::from(facing_gap(hero.facing.brads, wanted));
    let turn = gap
        .saturating_sub(u32::from(ATTACK_ANGLE_BRADS))
        .div_ceil(TURN_RATE_BRADS);
    let distance_raw = isqrt(hero.pos.distance_squared(target.pos) as u64);
    let distance = distance_raw.div_ceil(u64::from(Fixed::ONE.raw as u32));
    let travel = distance.div_ceil(ATTACK_PROJECTILE_UNITS_PER_TICK as u64) as u32;
    hero.attack_interval
        .saturating_add(turn)
        .saturating_add(ATTACK_POINT_TICKS)
        .saturating_add(travel)
}

fn in_attack_reach(hero: &UnitView, target: &UnitView) -> bool {
    hero.pos
        .within(target.pos, hero.attack_range + hero.radius + target.radius)
}

fn in_attack_reach_with_leeway(hero: &UnitView, target: &UnitView) -> bool {
    hero.pos.within(
        target.pos,
        hero.attack_range + hero.radius + target.radius + Fixed::from_int(ATTACK_RANGE_LEEWAY),
    )
}

fn best_farm_raze(
    tracker: &StateTracker,
    space: &ActionSpace,
    hero: &UnitView,
) -> Option<(EntityIndex, usize)> {
    let mut best = None;
    for (index, candidate) in space.entity_candidates().iter().enumerate() {
        let creep = candidate.unit();
        if !hero.pos.within(creep.pos, Fixed::from_int(950)) {
            continue;
        }
        for (slot, ability) in hero.abilities.iter().enumerate() {
            let Some(reach) = raze_reach(ability.id) else {
                continue;
            };
            let damage = raze_damage(ability.level);
            if !space.allows(cast_none(slot))
                || hero.mana < ability.mana_cost.saturating_mul(2)
                || !raze_farm_target(hero, creep, damage)
            {
                continue;
            }
            let facing = if raze_contains(tracker, hero, creep, hero.facing.brads, reach) {
                hero.facing.brads
            } else {
                if !tactical_chase_safe(tracker, hero, creep.pos) {
                    continue;
                }
                facing_towards(hero.pos, predicted_position(tracker, creep))
            };
            if !raze_contains(tracker, hero, creep, facing, reach) {
                continue;
            }
            let (kills, error) = farm_raze_hits(tracker, hero, facing, reach, damage);
            assert!(kills > 0);
            let score = (
                std::cmp::Reverse(kills),
                error,
                facing_gap(hero.facing.brads, facing),
                creep.id,
                slot,
            );
            if best.is_none_or(|(previous, _, _)| score < previous) {
                best = Some((score, EntityIndex(index), slot));
            }
        }
    }
    best.map(|(_, target, slot)| (target, slot))
}

fn raze_farm_target(hero: &UnitView, creep: &UnitView, damage: i32) -> bool {
    creep.team != hero.team
        && (is_lane_creep(creep.kind) || creep.kind == UnitKind::CreepNeutral)
        && creep.hp > 0
        && creep.hp <= magical_damage(damage, creep.magic_resist)
        && (!in_attack_reach(hero, creep)
            || creep.hp > physical_damage(attack_damage_against(hero, creep), creep.armor))
}

fn raze_contains(
    tracker: &StateTracker,
    hero: &UnitView,
    target: &UnitView,
    facing: u16,
    reach: i32,
) -> bool {
    let radius = Fixed {
        raw: Fixed::from_int(SHADOWRAZE_RADIUS)
            .raw
            .saturating_sub(target.move_speed.raw.max(0) / 30)
            .max(0),
    };
    raze_center(hero.pos, facing, reach).within(predicted_position(tracker, target), radius)
}

fn farm_raze_hits(
    tracker: &StateTracker,
    hero: &UnitView,
    facing: u16,
    reach: i32,
    damage: i32,
) -> (usize, i64) {
    let center = raze_center(hero.pos, facing, reach);
    tracker.current().map_or((0, 0), |view| {
        view.units
            .iter()
            .filter(|creep| {
                raze_farm_target(hero, creep, damage)
                    && raze_contains(tracker, hero, creep, facing, reach)
            })
            .fold((0, 0), |(kills, error), creep| {
                (
                    kills + 1,
                    error.max(center.distance_squared(predicted_position(tracker, creep))),
                )
            })
    })
}

fn raze_reach(id: AbilityId) -> Option<i32> {
    SHADOWRAZES
        .iter()
        .find_map(|(raze, reach)| (*raze == id).then_some(*reach))
}

fn raze_damage(level: u8) -> i32 {
    SHADOWRAZE_DAMAGE[usize::from(level.clamp(1, 4) - 1)]
}

fn cast_none(slot: usize) -> StructuredAction {
    StructuredAction::Cast {
        unit: ControlledUnit::Hero,
        slot: AbilitySlot(slot as u8),
        target: ActionTarget::None,
    }
}

fn raze_center(position: Vec2, facing: u16, distance: i32) -> Vec2 {
    let ahead = position + heading_of(facing);
    point_along(position, ahead, Fixed::from_int(distance))
}

fn enemy_heroes(tracker: &StateTracker) -> impl Iterator<Item = &UnitView> {
    tracker.current().into_iter().flat_map(|view| {
        view.units.iter().filter(|unit| {
            unit.kind == UnitKind::Hero && unit.team != tracker.team() && unit.hp > 0
        })
    })
}

fn requiem_damage(level: u8, souls: u32) -> i32 {
    let index = usize::from(level.clamp(1, 3) - 1);
    REQUIEM_DAMAGE_PER_SOUL[index].saturating_mul(souls.min(i32::MAX as u32) as i32)
}

fn unsafe_tower_without_wave(tracker: &StateTracker, hero: &UnitView) -> bool {
    enemy_tower_danger(tracker, hero.pos, hero.radius) && !allied_creep_near(tracker, hero.pos, 750)
}

fn enemy_tower_danger(tracker: &StateTracker, position: Vec2, radius: Fixed) -> bool {
    tracker.current().is_some_and(|view| {
        view.units.iter().any(|unit| {
            unit.kind == UnitKind::Tower
                && unit.team != tracker.team()
                && unit.hp > 0
                && position.within(unit.pos, Fixed::from_int(700) + radius + unit.radius)
        })
    })
}

fn allied_creep_near(tracker: &StateTracker, position: Vec2, radius: i32) -> bool {
    tracker.current().is_some_and(|view| {
        view.units.iter().any(|unit| {
            unit.team == tracker.team()
                && is_lane_creep(unit.kind)
                && unit.hp > 0
                && position.within(unit.pos, Fixed::from_int(radius))
        })
    })
}

fn own_fountain(tracker: &StateTracker) -> Option<Vec2> {
    tracker.current()?.units.iter().find_map(|unit| {
        (unit.kind == UnitKind::Fountain && unit.team == tracker.team()).then_some(unit.pos)
    })
}

fn best_safe_point(
    space: &ActionSpace,
    tracker: &StateTracker,
    mask: &[bool],
    wanted: Vec2,
    require_progress: bool,
) -> Option<PointIndex> {
    let hero = tracker.own_hero()?;
    let current = hero.pos.distance_squared(wanted);
    space
        .point_candidates()
        .iter()
        .enumerate()
        .filter(|(index, point)| {
            mask.get(*index) == Some(&true)
                && (!require_progress || point.position.distance_squared(wanted) < current)
                && tower_corridor_safe(tracker, hero, point.position, true)
                && (!enemy_tower_danger(tracker, point.position, hero.radius)
                    || allied_creep_near(tracker, point.position, 750))
        })
        .min_by_key(|(_, point)| point.position.distance_squared(wanted))
        .map(|(index, _)| PointIndex(index))
}

fn nearest_visible_creep(tracker: &StateTracker, position: Vec2) -> Option<&UnitView> {
    tracker
        .current()?
        .units
        .iter()
        .filter(|unit| is_lane_creep(unit.kind) && unit.hp > 0)
        .min_by_key(|unit| (position.distance_squared(unit.pos), unit.id))
}

fn enemy_objective(tracker: &StateTracker, position: Vec2) -> Option<Vec2> {
    tracker
        .current()?
        .units
        .iter()
        .filter(|unit| {
            unit.team != tracker.team()
                && unit.hp > 0
                && matches!(
                    unit.kind,
                    UnitKind::Tower | UnitKind::Ancient | UnitKind::Barracks
                )
        })
        .min_by_key(|unit| (position.distance_squared(unit.pos), unit.id))
        .map(|unit| unit.pos)
}

fn is_lane_creep(kind: UnitKind) -> bool {
    matches!(
        kind,
        UnitKind::CreepMelee
            | UnitKind::CreepFlagbearer
            | UnitKind::CreepRanged
            | UnitKind::CreepSiege
    )
}

fn reject_note(notes: &mut [Option<OrderNote>; ORDER_NOTE_LIMIT], sequence: u32) -> bool {
    let mut changed = false;
    for note in notes {
        if note.is_some_and(|note| note.sequence == sequence) {
            *note = None;
            changed = true;
        }
    }
    changed
}

fn is_courier_errand_order(tracker: &StateTracker, order: Order) -> bool {
    let Order::Cast { slot, .. } = order else {
        return false;
    };
    tracker
        .own_courier()
        .and_then(|courier| courier.abilities.get(usize::from(slot.0)))
        .is_some_and(|ability| matches!(ability.id, COURIER_TAKE_STASH | COURIER_DELIVER))
}

const fn is_item_use_order(order: Order) -> bool {
    matches!(order, Order::Use { .. })
}

const fn is_notable_order(order: Order) -> bool {
    matches!(
        order,
        Order::Move { .. }
            | Order::Attack { .. }
            | Order::Cast { .. }
            | Order::Use { .. }
            | Order::Put { .. }
            | Order::Take { .. }
    )
}

fn facing_towards(from: Vec2, to: Vec2) -> u16 {
    let dx = i64::from(to.x.raw) - i64::from(from.x.raw);
    let dy = i64::from(to.y.raw) - i64::from(from.y.raw);
    if dx == 0 && dy == 0 {
        return 0;
    }
    let (absolute_x, absolute_y) = (dx.abs(), dy.abs());
    let slope = if absolute_x >= absolute_y {
        (absolute_y << 13) / absolute_x
    } else {
        (absolute_x << 13) / absolute_y
    };
    let octant = match (dx >= 0, dy >= 0, absolute_x >= absolute_y) {
        (true, true, true) => slope,
        (true, true, false) => 16_384 - slope,
        (false, true, false) => 16_384 + slope,
        (false, true, true) => 32_768 - slope,
        (false, false, true) => 32_768 + slope,
        (false, false, false) => 49_152 - slope,
        (true, false, false) => 49_152 + slope,
        (true, false, true) => 65_536 - slope,
    };
    (octant & 0xffff) as u16
}

fn facing_gap(one: u16, other: u16) -> u16 {
    let clockwise = one.wrapping_sub(other);
    let counterclockwise = other.wrapping_sub(one);
    clockwise.min(counterclockwise)
}

fn heading_of(facing: u16) -> Vec2 {
    let brads = i32::from(facing);
    let slope = brads % 8_192;
    let (x, y) = match brads / 8_192 {
        0 => (8_192, slope),
        1 => (8_192 - slope, 8_192),
        2 => (-slope, 8_192),
        3 => (-8_192, 8_192 - slope),
        4 => (-8_192, -slope),
        5 => (-(8_192 - slope), -8_192),
        6 => (slope, -8_192),
        _ => (8_192, -(8_192 - slope)),
    };
    Vec2::from_ints(x, y)
}

fn point_along(from: Vec2, towards: Vec2, distance: Fixed) -> Vec2 {
    let x = i64::from(towards.x.raw) - i64::from(from.x.raw);
    let y = i64::from(towards.y.raw) - i64::from(from.y.raw);
    let span = isqrt((x * x + y * y) as u64) as i64;
    if span == 0 {
        return from;
    }
    Vec2 {
        x: Fixed {
            raw: from
                .x
                .raw
                .saturating_add((x * i64::from(distance.raw) / span) as i32),
        },
        y: Fixed {
            raw: from
                .y
                .raw
                .saturating_add((y * i64::from(distance.raw) / span) as i32),
        },
    }
}

fn isqrt(value: u64) -> u64 {
    let mut remainder = value;
    let mut root = 0_u64;
    let mut bit = 1_u64 << 62;
    for _ in 0..32 {
        if remainder >= root.saturating_add(bit) {
            remainder -= root + bit;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }
        bit >>= 2;
    }
    root
}
