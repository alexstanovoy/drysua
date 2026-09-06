use bota_proto::{
    AbilityId, EffectId, EntityId, EventKind, Fixed, ItemId, ItemSlot, ItemView, Team, UnitKind,
    UnitView,
};

use crate::{
    ActionSpace, ActionTarget, ControlledUnit, MAX_ABILITY_SLOTS, MAX_RECENT_EVENTS, MAX_SEATS,
    MAX_TRACKED_ENTITIES, PointIndex, PointSource, ShopIndex, StateTracker, StructuredAction,
};

const ACTIVE_ITEM_SLOTS: usize = 6;
const CLARITY: ItemId = ItemId(1);
const HEALING_SALVE: ItemId = ItemId(2);
const MAGIC_STICK: ItemId = ItemId(35);
const MAGIC_WAND: ItemId = ItemId(36);
const MENDING: EffectId = EffectId(1);
const POWER_TREADS: ItemId = ItemId(29);
const QUELLING_BLADE: ItemId = ItemId(5);
const RECENT_CAST_TICKS: u32 = 300;
const RECENT_DAMAGE_TICKS: u32 = 90;
const RESTORE_PER_CHARGE: i32 = 15;
const TANGO: ItemId = ItemId(7);

/// Once-only purchases: Wraith Band, Tango, Boots, optional Stick, Gloves, Belt.
/// Teacher owns sent/rejected bookkeeping; Boots, Gloves and Belt assemble as Treads.
pub(crate) const ECONOMY_PLAN: [ItemId; 6] = [
    ItemId(33),
    TANGO,
    ItemId(0),
    MAGIC_STICK,
    ItemId(19),
    ItemId(13),
];

const _: () = assert!(ECONOMY_PLAN.len() <= ACTIVE_ITEM_SLOTS);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EnemyCooldowns {
    id: EntityId,
    abilities: [Option<(AbilityId, u32)>; MAX_ABILITY_SLOTS],
    confirmed_cast: Option<u32>,
}

/// Decision-local visible cooldown baselines and recently confirmed enemy casts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EconomyObservation {
    tick: Option<u32>,
    heroes: [Option<EnemyCooldowns>; MAX_SEATS],
}

impl EconomyObservation {
    pub(crate) const fn new() -> Self {
        Self {
            tick: None,
            heroes: [None; MAX_SEATS],
        }
    }

    /// Observes a completed snapshot/event tick before economy selection; repeated ticks are inert.
    pub(crate) fn observe(&mut self, tracker: &StateTracker) {
        let Some(view) = tracker.current() else {
            return;
        };
        assert!(view.units.len() <= MAX_TRACKED_ENTITIES);
        if self.tick == Some(view.tick) {
            return;
        }
        assert!(self.tick.is_none_or(|tick| tick < view.tick));
        let mut heroes = [None; MAX_SEATS];
        for (index, enemy) in view
            .units
            .iter()
            .filter(|unit| {
                unit.kind == UnitKind::Hero
                    && unit.hp > 0
                    && unit.team != tracker.team()
                    && unit.team != Team::Neutral
            })
            .take(MAX_SEATS)
            .enumerate()
        {
            assert!(enemy.abilities.len() <= MAX_ABILITY_SLOTS);
            let previous = self
                .heroes
                .iter()
                .flatten()
                .find(|hero| hero.id == enemy.id);
            let confirmed_cast = previous.and_then(|previous| {
                confirmed_cast_tick(
                    tracker,
                    enemy,
                    previous,
                    self.tick.expect("prior observation"),
                )
                .or(previous.confirmed_cast)
            });
            heroes[index] = Some(EnemyCooldowns {
                id: enemy.id,
                abilities: std::array::from_fn(|slot| {
                    enemy
                        .abilities
                        .get(slot)
                        .map(|ability| (ability.id, ability.cooldown_left))
                }),
                confirmed_cast,
            });
        }
        self.heroes = heroes;
        self.tick = Some(view.tick);
    }

    fn enemy_cast_available(&self, tracker: &StateTracker, hero: &UnitView) -> bool {
        let Some(view) = tracker.current() else {
            return false;
        };
        assert!(self.tick.is_none_or(|tick| tick == view.tick));
        self.heroes.iter().flatten().any(|enemy| {
            enemy
                .confirmed_cast
                .is_some_and(|tick| recent(view.tick, tick, RECENT_CAST_TICKS))
                && tracker.entity(enemy.id).is_some_and(|enemy| {
                    enemy.visible && hero.pos.within(enemy.unit.pos, Fixed::from_int(1_200))
                })
        })
    }
}

/// Selects one affordable planned purchase using the caller's once-only purchase journal.
pub(crate) fn select_purchase(
    tracker: &StateTracker,
    space: &ActionSpace,
    bought_once: &[bool; ECONOMY_PLAN.len()],
    observation: &EconomyObservation,
) -> Option<StructuredAction> {
    let hero = tracker.own_hero().filter(|hero| hero.hp > 0)?;
    assert_eq!(space.tick(), tracker.current()?.tick);
    assert!(hero.items.len() <= 9);
    for (index, wanted) in ECONOMY_PLAN.into_iter().enumerate() {
        if bought_once[index] || purchase_satisfied(tracker, wanted) {
            continue;
        }
        if wanted == MAGIC_STICK && !observation.enemy_cast_available(tracker, hero) {
            continue;
        }
        let item = space
            .shop_candidates()
            .iter()
            .position(|item| item.item == wanted)?;
        let action = StructuredAction::Buy {
            unit: ControlledUnit::Hero,
            item: ShopIndex(item),
        };
        return space.allows(action).then_some(action);
    }
    None
}

/// Selects legal regeneration or charge restoration without replacing an active health drink.
/// `emergency` is the caller's visible combat danger assessment, not a health threshold alone.
pub(crate) fn select_sustain(
    tracker: &StateTracker,
    space: &ActionSpace,
    emergency: bool,
) -> Option<StructuredAction> {
    let hero = tracker
        .own_hero()
        .filter(|hero| hero.hp > 0 && hero.max_hp > 0)?;
    assert_eq!(space.tick(), tracker.current()?.tick);
    assert!(hero.items.len() <= 9);
    for wanted in [MAGIC_WAND, MAGIC_STICK, TANGO, HEALING_SALVE, CLARITY] {
        for (slot, held) in hero.items.iter().take(ACTIVE_ITEM_SLOTS).enumerate() {
            let Some(held) = held.as_ref().filter(|item| item.id == wanted) else {
                continue;
            };
            let Some(target) = sustain_target(tracker, space, hero, held, slot, emergency) else {
                continue;
            };
            let action = StructuredAction::Use {
                unit: ControlledUnit::Hero,
                slot: ItemSlot(slot as u8),
                target,
            };
            if space.allows(action) {
                return Some(action);
            }
        }
    }
    None
}

/// Raw attack damage with active Quelling bonuses for lane and neutral creeps, including denies.
pub(crate) fn attack_damage_against(source: &UnitView, target: &UnitView) -> i32 {
    assert!(source.items.len() <= 9);
    let base = source.attack_damage.max(0);
    let creep = matches!(
        target.kind,
        UnitKind::CreepMelee
            | UnitKind::CreepFlagbearer
            | UnitKind::CreepRanged
            | UnitKind::CreepSiege
            | UnitKind::CreepNeutral
    );
    let blades = if creep && source.kind != UnitKind::Courier {
        source
            .items
            .iter()
            .take(ACTIVE_ITEM_SLOTS)
            .flatten()
            .filter(|item| item.id == QUELLING_BLADE && item.mute_left == 0)
            .count()
    } else {
        0
    };
    assert!(blades <= ACTIVE_ITEM_SLOTS);
    base.saturating_add(18 * blades as i32)
}

fn purchase_satisfied(tracker: &StateTracker, item: ItemId) -> bool {
    holds_item(tracker, item)
        || matches!(item, ItemId(0) | ItemId(19) | ItemId(13)) && holds_item(tracker, POWER_TREADS)
        || item == MAGIC_STICK && holds_item(tracker, MAGIC_WAND)
}

/// Whether an item is held by the own hero, courier, or stash, including inert slots.
pub(crate) fn holds_item(tracker: &StateTracker, item: ItemId) -> bool {
    let hero = tracker
        .own_hero()
        .into_iter()
        .flat_map(|unit| unit.items.iter().take(9).flatten());
    let courier = tracker
        .own_courier()
        .into_iter()
        .flat_map(|unit| unit.items.iter().take(ACTIVE_ITEM_SLOTS).flatten());
    let stash = tracker
        .own_player()
        .and_then(|player| player.stash.as_ref())
        .into_iter()
        .flat_map(|slots| slots.iter().take(6).flatten());
    hero.chain(courier).chain(stash).any(|held| held.id == item)
}

fn confirmed_cast_tick(
    tracker: &StateTracker,
    enemy: &UnitView,
    previous: &EnemyCooldowns,
    previous_tick: u32,
) -> Option<u32> {
    let view = tracker.current()?;
    assert!(tracker.recent_events().len() <= MAX_RECENT_EVENTS);
    assert_eq!(previous.id, enemy.id);
    tracker
        .recent_events()
        .iter()
        .filter_map(|event| {
            let EventKind::AbilityCast { caster, ability } = event.kind else {
                return None;
            };
            if caster != enemy.id || event.tick <= previous_tick || event.tick > view.tick {
                return None;
            }
            let (_, before) = previous
                .abilities
                .iter()
                .flatten()
                .find(|(id, _)| *id == ability)?;
            let now = enemy.abilities.iter().find(|held| held.id == ability)?;
            // Learning emits AbilityCast too, but cannot increase an existing cooldown.
            (now.level > 0 && !now.passive && now.cooldown_left > *before).then_some(event.tick)
        })
        .min()
}

fn sustain_target(
    tracker: &StateTracker,
    space: &ActionSpace,
    hero: &UnitView,
    item: &ItemView,
    slot: usize,
    emergency: bool,
) -> Option<ActionTarget> {
    assert!(slot < ACTIVE_ITEM_SLOTS);
    assert_eq!(hero.items[slot], Some(*item));
    let health_missing = hero.max_hp.saturating_sub(hero.hp).max(0);
    let mana_missing = hero.max_mana.saturating_sub(hero.mana).max(0);
    match item.id {
        MAGIC_WAND | MAGIC_STICK => {
            let restore = i32::from(item.charges.unwrap_or(0)) * RESTORE_PER_CHARGE;
            let useful = health_missing.min(restore) + mana_missing.min(restore);
            (useful >= 90 || emergency && useful > 0).then_some(ActionTarget::None)
        }
        TANGO if health_missing >= 115 && !has_effect(hero, MENDING) => {
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
                .map(|(index, _)| ActionTarget::Point(PointIndex(index)))
        }
        HEALING_SALVE
            if health_missing >= 300 && !has_effect(hero, MENDING) && drink_safe(tracker, hero) =>
        {
            space.entity_index(hero.id).map(ActionTarget::Entity)
        }
        CLARITY
            if mana_missing >= 120
                && !has_effect(hero, EffectId(2))
                && drink_safe(tracker, hero) =>
        {
            space.entity_index(hero.id).map(ActionTarget::Entity)
        }
        _ => None,
    }
}

fn drink_safe(tracker: &StateTracker, hero: &UnitView) -> bool {
    let Some(view) = tracker.current() else {
        return false;
    };
    assert!(view.units.len() <= MAX_TRACKED_ENTITIES);
    assert!(tracker.recent_events().len() <= MAX_RECENT_EVENTS);
    let hurt = tracker.recent_events().iter().any(|event| {
        let EventKind::Damaged {
            source: Some(source),
            target,
            amount,
            ..
        } = event.kind
        else {
            return false;
        };
        target == hero.id
            && amount > 0
            && recent(view.tick, event.tick, RECENT_DAMAGE_TICKS)
            && damage_breaks_drink(tracker, source)
    });
    !hurt
        && !view.units.iter().any(|enemy| {
            if enemy.team == tracker.team()
                || enemy.team == Team::Neutral
                || enemy.hp <= 0
                || !matches!(enemy.kind, UnitKind::Hero | UnitKind::Tower)
            {
                return false;
            }
            let reach = Fixed {
                raw: enemy
                    .attack_range
                    .raw
                    .saturating_add(enemy.radius.raw)
                    .saturating_add(hero.radius.raw)
                    .max(Fixed::from_int(700).raw),
            };
            hero.pos.within(enemy.pos, reach)
        })
}

fn damage_breaks_drink(tracker: &StateTracker, source: EntityId) -> bool {
    tracker
        .entity(source)
        .is_none_or(|source| matches!(source.unit.kind, UnitKind::Hero | UnitKind::Tower))
}

fn has_effect(hero: &UnitView, effect: EffectId) -> bool {
    hero.effects
        .iter()
        .any(|held| held.id == effect && held.ticks_left.is_none_or(|ticks| ticks > 0))
}

fn recent(now: u32, tick: u32, limit: u32) -> bool {
    now.checked_sub(tick).is_some_and(|age| age <= limit)
}
