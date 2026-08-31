use bota_proto::{
    AbilityId, AbilitySlot, EffectId, EntityId, Fixed, ItemId, ItemSlot, ItemView, Order, Target,
    Team, UnitKind, UnitView, Vec2,
};

use crate::{
    ActionError, ActionSpace, ActionTarget, ControlledUnit, EntityIndex, EntityRelation,
    IssuedOrder, ItemReadiness, OrderPersistence, PointIndex, PointSource, StateTracker,
    StructuredAction, TOWN_PORTAL_SCROLL,
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
const CLARITY: ItemId = ItemId(1);
const HEALING_SALVE: ItemId = ItemId(2);
const TANGO: ItemId = ItemId(7);
const WRAITH_BAND: ItemId = ItemId(33);
const MAGIC_STICK: ItemId = ItemId(35);
const MAGIC_WAND: ItemId = ItemId(36);
const POWER_TREADS: ItemId = ItemId(29);
const BUILD_PLAN: [ItemId; 7] = [
    TANGO,
    HEALING_SALVE,
    CLARITY,
    TOWN_PORTAL_SCROLL,
    WRAITH_BAND,
    POWER_TREADS,
    MAGIC_WAND,
];
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
        }
    }

    /// Selects an action and returns the exact action space used to select it.
    pub fn decide(
        &mut self,
        tracker: &StateTracker,
        persistence: &OrderPersistence,
        readiness: &ItemReadiness,
    ) -> Result<(StructuredAction, ActionSpace), ActionError> {
        let space = ActionSpace::from_tracker_with_readiness(tracker, readiness)?;
        self.sync_purchases(tracker);
        self.sync_order_notes(tracker);
        let selected = self.priority_action(tracker, persistence, &space);
        if !space.allows(selected) {
            return Err(ActionError::InvalidSchema("teacher selected masked action"));
        }
        let decoded = space.decode(selected)?;
        let action = if decoded.is_some() && persistence.should_send(decoded).is_none() {
            StructuredAction::Continue
        } else {
            selected
        };
        Ok((action, space))
    }

    /// Records a sent order and the snapshot tick of the space that decoded it.
    pub fn note_sent(&mut self, sequence: u32, issued: IssuedOrder, tick: u32) {
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
    ) -> StructuredAction {
        if self.protects_channel(space) {
            return StructuredAction::Continue;
        }
        if let Some(action) = self
            .learn(tracker, space)
            .or_else(|| self.buy(tracker, space))
            .or_else(|| self.courier(tracker, space))
            .or_else(|| self.sustain(tracker, space))
        {
            return action;
        }
        if self.protects_useful_persistent_order(tracker, persistence, space) {
            return StructuredAction::Continue;
        }
        self.retreat(tracker, space)
            .or_else(|| self.attack_last_hit(tracker, space))
            .or_else(|| self.raze_last_hit(tracker, space))
            .or_else(|| self.deny(tracker, space))
            .or_else(|| self.raze_hero(tracker, space))
            .or_else(|| self.requiem(tracker, space))
            .or_else(|| self.harass(tracker, space))
            .or_else(|| self.attack_structure(tracker, space))
            .or_else(|| self.hold_lane(tracker, space))
            .or_else(|| self.safe_objective(tracker, space))
            .unwrap_or(StructuredAction::Continue)
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

    fn protects_channel(&self, space: &ActionSpace) -> bool {
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

    fn protects_useful_persistent_order(
        &self,
        tracker: &StateTracker,
        persistence: &OrderPersistence,
        space: &ActionSpace,
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
            Order::Attack {
                target: Target::None,
            } => useful_hold(tracker),
            Order::Attack {
                target: Target::Unit(target),
            } => self.continue_attack(tracker, space, target),
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
        let mask = space.buy_mask(ControlledUnit::Hero);
        for (plan_index, wanted) in BUILD_PLAN.iter().copied().enumerate() {
            let satisfied = if wanted == TOWN_PORTAL_SCROLL {
                holds_item(tracker, wanted)
            } else {
                self.bought_once[plan_index] || holds_item(tracker, wanted)
            };
            if satisfied {
                continue;
            }
            let shop_index = space
                .shop_candidates()
                .iter()
                .position(|candidate| candidate.item == wanted)?;
            return mask.get(shop_index).copied().unwrap_or(false).then_some(
                StructuredAction::Buy {
                    unit: ControlledUnit::Hero,
                    item: crate::ShopIndex(shop_index),
                },
            );
        }
        None
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
        if hero.max_hp <= 0 || hero.max_mana < 0 {
            return None;
        }
        let recently_hurt = recently_damaged(tracker, hero, 90);
        for item in [MAGIC_WAND, MAGIC_STICK, TANGO, HEALING_SALVE, CLARITY] {
            for (slot, held) in hero.items.iter().take(6).enumerate() {
                let Some(held) = held.as_ref().filter(|held| held.id == item) else {
                    continue;
                };
                let action = self.use_sustain(tracker, space, hero, held, slot, recently_hurt);
                if action.is_some() {
                    return action;
                }
            }
        }
        None
    }

    fn use_sustain(
        &self,
        tracker: &StateTracker,
        space: &ActionSpace,
        hero: &UnitView,
        held: &ItemView,
        slot: usize,
        recently_hurt: bool,
    ) -> Option<StructuredAction> {
        let hp_missing = hero.max_hp.saturating_sub(hero.hp);
        let mana_missing = hero.max_mana.saturating_sub(hero.mana);
        let target = match held.id {
            MAGIC_WAND | MAGIC_STICK
                if held.charges.unwrap_or(0) >= 3
                    && (ratio_at_most(hero.hp, hero.max_hp, 40)
                        || hp_missing.saturating_add(mana_missing) >= 120) =>
            {
                ActionTarget::None
            }
            TANGO if ratio_at_most(hero.hp, hero.max_hp, 70) && hp_missing >= 115 => {
                ActionTarget::Point(first_tree_target(space, slot)?)
            }
            HEALING_SALVE
                if !recently_hurt
                    && !enemy_near(tracker, hero.pos, 700)
                    && ratio_at_most(hero.hp, hero.max_hp, 35)
                    && hp_missing >= 300
                    && !has_effect(hero, EffectId(1)) =>
            {
                ActionTarget::Entity(own_hero_index(space)?)
            }
            CLARITY
                if !recently_hurt
                    && !enemy_near(tracker, hero.pos, 700)
                    && ratio_at_most(hero.mana, hero.max_mana, 25)
                    && mana_missing >= 120
                    && !has_effect(hero, EffectId(2)) =>
            {
                ActionTarget::Entity(own_hero_index(space)?)
            }
            _ => return None,
        };
        let action = StructuredAction::Use {
            unit: ControlledUnit::Hero,
            slot: ItemSlot(slot as u8),
            target,
        };
        space.allows(action).then_some(action)
    }

    fn retreat(&self, tracker: &StateTracker, space: &ActionSpace) -> Option<StructuredAction> {
        let hero = tracker.own_hero()?;
        let critical = ratio_at_most(hero.hp, hero.max_hp, 25);
        let lethal = visible_pressure(tracker, hero) >= hero.hp.max(0);
        let tower = unsafe_tower_without_wave(tracker, hero);
        if !critical && !lethal && !tower {
            return None;
        }
        let fountain = own_fountain(tracker)?;
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

    fn raze_last_hit(
        &self,
        tracker: &StateTracker,
        space: &ActionSpace,
    ) -> Option<StructuredAction> {
        let hero = tracker.own_hero()?;
        let mut best: Option<(usize, usize)> = None;
        for (slot, ability) in hero.abilities.iter().enumerate() {
            let Some(reach) = raze_reach(ability.id) else {
                continue;
            };
            let action = cast_none(slot);
            if !space.allows(action) || hero.mana < ability.mana_cost.saturating_mul(2) {
                continue;
            }
            let damage = raze_damage(ability.level);
            let kills = raze_creep_kills(tracker, hero, reach, damage);
            if kills > 0
                && best.is_none_or(|(best_kills, best_slot)| {
                    kills > best_kills || kills == best_kills && slot < best_slot
                })
            {
                best = Some((kills, slot));
            }
        }
        best.map(|(_, slot)| cast_none(slot))
    }

    fn deny(&self, tracker: &StateTracker, space: &ActionSpace) -> Option<StructuredAction> {
        let target = best_attack_creep(tracker, space, EntityRelation::Allied, true)?;
        Some(StructuredAction::AttackUnit {
            unit: ControlledUnit::Hero,
            target,
        })
    }

    fn raze_hero(&self, tracker: &StateTracker, space: &ActionSpace) -> Option<StructuredAction> {
        let hero = tracker.own_hero()?;
        for (slot, ability) in hero.abilities.iter().enumerate() {
            let Some(reach) = raze_reach(ability.id) else {
                continue;
            };
            let action = cast_none(slot);
            if !space.allows(action) || hero.mana < ability.mana_cost.saturating_add(100) {
                continue;
            }
            let center = raze_center(hero.pos, hero.facing.brads, reach);
            if enemy_heroes(tracker)
                .any(|hero| center.within(hero.pos, Fixed::from_int(SHADOWRAZE_RADIUS)))
            {
                return Some(action);
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
    healthy
        && pressure_safe
        && !enemy_tower_danger(tracker, hero.pos, hero.radius)
        && !enemy_tower_danger(tracker, target, hero.radius)
}

fn useful_hold(tracker: &StateTracker) -> bool {
    let Some(hero) = tracker.own_hero() else {
        return false;
    };
    !ratio_at_most(hero.hp, hero.max_hp, 25)
        && visible_pressure(tracker, hero) < hero.hp.max(0)
        && !enemy_tower_danger(tracker, hero.pos, hero.radius)
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

fn first_tree_target(space: &ActionSpace, slot: usize) -> Option<PointIndex> {
    let mask = space.use_target_mask(ControlledUnit::Hero, ItemSlot(slot as u8))?;
    space
        .point_candidates()
        .iter()
        .enumerate()
        .find(|(index, point)| {
            mask.points().get(*index) == Some(&true)
                && matches!(
                    point.source,
                    PointSource::StaticTree | PointSource::PlantedTree
                )
        })
        .map(|(index, _)| PointIndex(index))
}

fn own_hero_index(space: &ActionSpace) -> Option<EntityIndex> {
    space
        .entity_candidates()
        .iter()
        .position(|candidate| {
            candidate.relation == EntityRelation::Own && candidate.kind == UnitKind::Hero
        })
        .map(EntityIndex)
}

fn holds_item(tracker: &StateTracker, wanted: ItemId) -> bool {
    let hero = tracker
        .own_hero()
        .into_iter()
        .flat_map(|unit| unit.items.iter().flatten());
    let courier = tracker
        .own_courier()
        .into_iter()
        .flat_map(|unit| unit.items.iter().flatten());
    let stash = tracker
        .own_player()
        .and_then(|player| player.stash.as_ref())
        .into_iter()
        .flat_map(|slots| slots.iter().flatten());
    hero.chain(courier)
        .chain(stash)
        .any(|item| item.id == wanted)
}

fn recently_damaged(tracker: &StateTracker, hero: &UnitView, age: u32) -> bool {
    let Some(tick) = tracker.current().map(|view| view.tick) else {
        return false;
    };
    tracker
        .entity(hero.id)
        .and_then(|entity| entity.last_damage_taken)
        .is_some_and(|damage| tick.saturating_sub(damage.tick) <= age)
}

fn has_effect(unit: &UnitView, effect: EffectId) -> bool {
    unit.effects.iter().any(|active| active.id == effect)
}

fn ratio_at_most(value: i32, maximum: i32, percent: i32) -> bool {
    maximum > 0 && i64::from(value) * 100 <= i64::from(maximum) * i64::from(percent)
}

fn ratio_below(value: i32, maximum: i32, percent: i32) -> bool {
    maximum <= 0 || i64::from(value) * 100 < i64::from(maximum) * i64::from(percent)
}

fn enemy_near(tracker: &StateTracker, position: Vec2, radius: i32) -> bool {
    tracker.current().is_some_and(|view| {
        view.units.iter().any(|unit| {
            unit.team != tracker.team()
                && unit.team != Team::Neutral
                && unit.hp > 0
                && position.within(unit.pos, Fixed::from_int(radius))
        })
    })
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
    let damage = physical_damage(hero.attack_damage, Fixed::ZERO);
    let mask = space.attack_entity_mask(ControlledUnit::Hero);
    space
        .entity_candidates()
        .iter()
        .enumerate()
        .filter_map(|(index, candidate)| {
            let unit = candidate.unit();
            if mask.get(index) != Some(&true)
                || candidate.relation != relation
                || !is_lane_creep(candidate.kind)
                || !in_attack_reach(hero, unit)
                || (deny && unit.hp.saturating_mul(2) >= unit.max_hp)
            {
                return None;
            }
            let landing = attack_landing_ticks(hero, unit);
            let predicted = predicted_hp(tracker, unit, landing);
            let hit = physical_damage(hero.attack_damage, unit.armor).min(damage);
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

fn raze_creep_kills(tracker: &StateTracker, hero: &UnitView, reach: i32, damage: i32) -> usize {
    let center = raze_center(hero.pos, hero.facing.brads, reach);
    tracker.current().map_or(0, |view| {
        view.units
            .iter()
            .filter(|unit| {
                unit.team != tracker.team()
                    && is_lane_creep(unit.kind)
                    && unit.pos.within(center, Fixed::from_int(SHADOWRAZE_RADIUS))
                    && unit.hp > 0
                    && unit.hp <= magical_damage(damage, unit.magic_resist)
                    && unit.hp > physical_damage(hero.attack_damage, unit.armor)
            })
            .count()
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
                && !enemy_tower_danger(tracker, point.position, hero.radius)
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
