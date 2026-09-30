use bota_proto::{
    AbilityId, AbilitySlot, AbilityView, Aim, Attribute, Attributes, EntityId, HeroId, ItemId,
    ItemSlot, ItemView, LootView, MapId, MatchInfo, Order, PlayerView, ShopEntry, SlotId, Target,
    Team, UnitKind, UnitView, Vec2, WorldView,
};

use super::fixtures;
use crate::{
    ActionError, ActionKind, ActionSpace, ActionTarget, ControlledUnit, EntityIndex, LootIndex,
    PointIndex, PutPointTarget, SHADOW_FIEND, ShopIndex, StateTracker, StructuredAction,
    UNIT_TOKENS,
};

const HERO: EntityId = EntityId {
    idx: 10,
    generation: 1,
};
const COURIER: EntityId = EntityId {
    idx: 11,
    generation: 1,
};
const ENEMY: EntityId = EntityId {
    idx: 20,
    generation: 1,
};
const LOOT: EntityId = EntityId {
    idx: 50,
    generation: 1,
};

#[test]
fn public_decode_preserves_action_indices_wire_orders_and_rejects_injected_indices() {
    let space = space(&match_info(), world_view(1));
    assert_eq!(space.decode(StructuredAction::Continue), Ok(None));
    assert_eq!(ActionKind::COUNT, 16);
    let cases = movement_orders(&space)
        .into_iter()
        .chain(ability_orders())
        .chain(inventory_orders(&space));
    for (index, (action, order)) in cases.enumerate() {
        assert_eq!(action.kind().index(), index + 1);
        assert_eq!(ActionKind::from_index(index + 1), Some(action.kind()));
        assert_gate(&space, action, Some(order));
    }
    assert_eq!(ActionKind::from_index(16), None);
    let issued = space
        .decode(StructuredAction::Stop {
            unit: ControlledUnit::Courier,
        })
        .expect("decode")
        .expect("order");
    assert_eq!(issued.unit, Some(COURIER));
    let error = space
        .decode(StructuredAction::AttackUnit {
            unit: ControlledUnit::Hero,
            target: EntityIndex(UNIT_TOKENS),
        })
        .expect_err("injected index");
    assert_eq!(
        error.to_string(),
        format!(
            "entity target index 96 is outside candidate count {}",
            space.entity_candidates().len()
        )
    );
}

fn movement_orders(space: &ActionSpace) -> [(StructuredAction, Order); 6] {
    let unit = ControlledUnit::Hero;
    let point = PointIndex(
        space
            .point_candidates()
            .iter()
            .position(|p| p.walkable)
            .expect("point"),
    );
    let position = Target::Pos(space.point_candidates()[point.0].position);
    let courier = space.entity_index(COURIER).expect("courier");
    let enemy = space.entity_index(ENEMY).expect("enemy");
    [
        (
            StructuredAction::Stop { unit },
            Order::Move {
                target: Target::None,
            },
        ),
        (
            StructuredAction::MovePoint { unit, point },
            Order::Move { target: position },
        ),
        (
            StructuredAction::FollowUnit {
                unit,
                target: courier,
            },
            Order::Move {
                target: Target::Unit(COURIER),
            },
        ),
        (
            StructuredAction::Hold { unit },
            Order::Attack {
                target: Target::None,
            },
        ),
        (
            StructuredAction::AttackMovePoint { unit, point },
            Order::Attack { target: position },
        ),
        (
            StructuredAction::AttackUnit {
                unit,
                target: enemy,
            },
            Order::Attack {
                target: Target::Unit(ENEMY),
            },
        ),
    ]
}

fn ability_orders() -> [(StructuredAction, Order); 2] {
    let unit = ControlledUnit::Hero;
    let ability = AbilitySlot(0);
    let slot = ItemSlot(0);
    [
        (
            StructuredAction::Cast {
                unit,
                slot: ability,
                target: ActionTarget::None,
            },
            Order::Cast {
                slot: ability,
                target: Target::None,
            },
        ),
        (
            StructuredAction::Use {
                unit,
                slot,
                target: ActionTarget::None,
            },
            Order::Use {
                slot,
                target: Target::None,
            },
        ),
    ]
}

fn inventory_orders(space: &ActionSpace) -> [(StructuredAction, Order); 7] {
    let unit = ControlledUnit::Hero;
    let slot = ItemSlot(0);
    let ability = AbilitySlot(0);
    let courier = space.entity_index(COURIER).expect("courier");
    [
        (
            StructuredAction::PutPoint {
                unit,
                source: slot,
                target: PutPointTarget::Underfoot,
            },
            Order::Put {
                slot,
                target: Target::None,
            },
        ),
        (
            StructuredAction::PutUnit {
                unit,
                source: slot,
                target: courier,
            },
            Order::Put {
                slot,
                target: Target::Unit(COURIER),
            },
        ),
        (
            StructuredAction::Take {
                unit,
                loot: LootIndex(0),
            },
            Order::Take {
                target: Target::Unit(LOOT),
            },
        ),
        (
            StructuredAction::Buy {
                unit,
                item: ShopIndex(0),
            },
            Order::Buy { item: ItemId(0) },
        ),
        (StructuredAction::Sell { unit, slot }, Order::Sell { slot }),
        (
            StructuredAction::Swap {
                unit,
                from: slot,
                to: ItemSlot(8),
            },
            Order::Swap {
                from: slot,
                to: ItemSlot(8),
            },
        ),
        (
            StructuredAction::Learn { slot: ability },
            Order::Learn { slot: ability },
        ),
    ]
}

#[test]
fn bounded_entity_selection_excludes_fog_and_retains_controlled_bodies() {
    let mut tracker = tracker_with_info_and_view(&match_info(), world_view(1));
    let mut view = world_view(2);
    view.units.retain(|unit| unit.id != ENEMY);
    for index in 100..220 {
        view.units.push(unit(
            EntityId {
                idx: index,
                generation: 1,
            },
            UnitKind::CreepMelee,
            Team::Dire,
            3_000 + index as i32,
            3_000,
        ));
    }
    tracker.observe_snapshot(&view).expect("crowded snapshot");
    let space = ActionSpace::from_tracker(&tracker).expect("bounded space");
    assert_eq!(space.entity_candidates().len(), UNIT_TOKENS);
    assert!(space.entity_index(ENEMY).is_none());
    for id in [HERO, COURIER] {
        assert!(space.entity_index(id).is_some());
    }
}

#[test]
fn public_range_and_ownership_masks_keep_exact_boundaries() {
    for (range, allowed) in [(299, false), (300, true)] {
        let mut view = world_view(1);
        view.units[0].abilities[1].range = range;
        view.units[0].items[0].as_mut().expect("item").for_sale = allowed;
        let space = space(&match_info(), view);
        let target = space.entity_index(ENEMY).expect("enemy");
        assert_gate(
            &space,
            StructuredAction::Cast {
                unit: ControlledUnit::Hero,
                slot: AbilitySlot(1),
                target: ActionTarget::Entity(target),
            },
            allowed.then_some(Order::Cast {
                slot: AbilitySlot(1),
                target: Target::Unit(ENEMY),
            }),
        );
        assert_gate(
            &space,
            StructuredAction::Sell {
                unit: ControlledUnit::Hero,
                slot: ItemSlot(0),
            },
            allowed.then_some(Order::Sell { slot: ItemSlot(0) }),
        );
    }
}

#[test]
fn raze_masks_allow_none_any_point_and_only_live_hostile_entities_within_reach_plus_minus_radius() {
    for (distance, near, far) in [(300, true, false), (451, true, true), (452, false, true)] {
        let mut view = world_view(1);
        view.units[0].abilities = crate::raze_aim::SHADOWRAZES
            .map(|(id, reach)| AbilityView {
                id,
                ..ability(Aim::Own, reach)
            })
            .to_vec();
        let enemy = view.units.iter_mut().find(|unit| unit.id == ENEMY);
        enemy.expect("enemy").pos = Vec2::from_ints(2_000 + distance, 2_000);
        let space = space(&match_info(), view);
        let enemy = ActionTarget::Entity(space.entity_index(ENEMY).expect("enemy"));
        let courier = ActionTarget::Entity(space.entity_index(COURIER).expect("courier"));
        for (slot, allowed) in [(0, near), (2, far)] {
            let mask = space
                .cast_target_mask(ControlledUnit::Hero, AbilitySlot(slot))
                .expect("raze slot");
            assert!(mask.allows_none());
            assert!(mask.points().iter().all(|allowed| *allowed));
            assert!(!mask.allows(courier));
            assert_eq!(mask.allows(enemy), allowed, "slot {slot} at {distance}");
        }
    }
}

#[test]
fn missing_snapshot_and_invalid_recipe_schemas_report_specific_errors() {
    let tracker = StateTracker::new(SlotId(0), &match_info()).expect("tracker");
    assert_eq!(
        ActionSpace::from_tracker(&tracker)
            .err()
            .expect("snapshot required")
            .to_string(),
        "action space requires a validated snapshot"
    );
    for (parts, message) in [
        ([vec![ItemId(1)], vec![ItemId(0)]], "cyclic shop recipe"),
        ([vec![ItemId(99)], vec![]], "unknown recipe component"),
    ] {
        let mut info = match_info();
        for (entry, parts) in info.shop.iter_mut().zip(parts) {
            entry.components = parts;
        }
        let tracker = tracker_with_info_and_view(&info, world_view(1));
        assert_eq!(
            ActionSpace::from_tracker(&tracker).err(),
            Some(ActionError::InvalidSchema(message))
        );
    }
}

#[cfg(feature = "builtin")]
#[test]
fn native_upgrade_trace_rejects_underpayment_then_executes_the_decoded_leaf() {
    let (mut world, info) = wraith_upgrade_world(209);
    let rejected = space(&info, world.view(Team::Radiant));
    assert_gate(&rejected, buy_action(&rejected, ItemId(33)), None);
    world.seats[0].gold = 210;
    let ready = space(&info, world.view(Team::Radiant));
    let action = buy_action(&ready, ItemId(33));
    assert_gate(&ready, action, Some(Order::Buy { item: ItemId(39) }));
    let issued = ready
        .decode(action)
        .expect("decode")
        .expect("leaf purchase");
    assert_eq!(
        world.validate_order(SlotId(0), issued.unit, &issued.order),
        Ok(())
    );
    world.advance(&[bota_server::game::Command {
        slot: SlotId(0),
        unit: issued.unit,
        order: issued.order,
    }]);
    assert_eq!(world.seats[0].gold, 0);
    let view = world.view(Team::Radiant);
    let hero = view.players[0].unit.expect("hero");
    let items = &view
        .units
        .iter()
        .find(|unit| unit.id == hero)
        .expect("hero body")
        .items;
    assert_eq!(
        items
            .iter()
            .flatten()
            .map(|item| item.id)
            .collect::<Vec<_>>(),
        [ItemId(33)]
    );
}

#[cfg(feature = "builtin")]
fn wraith_upgrade_world(gold: i32) -> (bota_server::game::World, MatchInfo) {
    let config = bota_server::game::MatchConfig {
        match_id: 92_009_002,
        master_key: [0; 32],
        picks: match_info().picks,
        map: MapId(1),
        tick_rate: 30,
        mode: bota_proto::TickMode::Lockstep,
        ack_timeout_ticks: 30,
        cheats: false,
        spawn_modifiers: Vec::new(),
    };
    let mut world = bota_server::game::World::for_match(&config, config.rng());
    for item in [ItemId(9), ItemId(11)] {
        assert_eq!(
            world.validate_order(SlotId(0), None, &Order::Buy { item }),
            Ok(())
        );
        assert!(world.buy(SlotId(0), item, &mut Vec::new()));
    }
    world.step();
    world.seats[0].gold = gold;
    (world, config.info())
}

#[cfg(feature = "builtin")]
fn buy_action(space: &ActionSpace, item: ItemId) -> StructuredAction {
    let index = space
        .shop_candidates()
        .iter()
        .position(|candidate| candidate.item == item)
        .expect("item");
    StructuredAction::Buy {
        unit: ControlledUnit::Hero,
        item: ShopIndex(index),
    }
}

pub(super) fn assert_gate(space: &ActionSpace, action: StructuredAction, order: Option<Order>) {
    assert_eq!(space.allows(action), order.is_some(), "{action:?}");
    if let Some(order) = order {
        let issued = space.decode(action).expect("decode").expect("wire order");
        assert_eq!(issued.unit, None);
        assert_eq!(issued.order, order);
    } else {
        let error = space.decode(action).expect_err("masked action");
        assert_eq!(error, ActionError::NotAllowed(action.kind()));
        assert_eq!(
            error.to_string(),
            format!(
                "action {:?} is masked by the current action space",
                action.kind()
            )
        );
    }
}

pub(super) fn space(info: &MatchInfo, view: WorldView) -> ActionSpace {
    ActionSpace::from_tracker(&tracker_with_info_and_view(info, view)).expect("space")
}

pub(super) fn tracker_with_info_and_view(info: &MatchInfo, view: WorldView) -> StateTracker {
    let mut tracker = StateTracker::new(SlotId(0), info).expect("tracker");
    tracker.observe_snapshot(&view).expect("snapshot");
    tracker
}

pub(super) fn match_info() -> MatchInfo {
    fixtures::MatchInfoFixture::new(1, MapId(0), fixtures::two_seat_picks(Team::Radiant))
        .pregame_ticks(90)
        .terrain_cells(128)
        .terrain_rle(vec![(16_384, 0x80)])
        .shop(shop_entries(&[(0, 50, &[]), (1, 700, &[])]))
        .build()
}

pub(super) fn world_view(tick: u32) -> WorldView {
    let mut hero = unit(HERO, UnitKind::Hero, Team::Radiant, 2_000, 2_000);
    hero.hero = Some(SHADOW_FIEND);
    hero.owner = Some(SlotId(0));
    hero.mana = 500;
    hero.max_mana = 500;
    hero.level = 6;
    hero.abilities = vec![
        AbilityView {
            can_level: true,
            ..ability(Aim::Own, 0)
        },
        ability(Aim::Unit, 300),
        ability(Aim::Point, 1_200),
        ability(Aim::Tree, 1_200),
    ];
    hero.items = vec![None; 9];
    hero.items[0] = Some(ItemView {
        for_sale: true,
        ..item(Some(Aim::Own), 0)
    });
    let mut courier = unit(COURIER, UnitKind::Courier, Team::Radiant, 2_100, 2_000);
    courier.owner = Some(SlotId(0));
    courier.items = vec![None; 6];
    courier.attack_damage = 0;
    let own = own_player();
    let enemy = PlayerView {
        slot: SlotId(1),
        team: Team::Dire,
        unit: Some(EntityId {
            idx: 777,
            generation: 4,
        }),
        level: 1,
        gold: None,
        stash: None,
        ..own.clone()
    };
    WorldView {
        tick,
        viewer: Some(Team::Radiant),
        units: vec![
            hero,
            courier,
            unit(ENEMY, UnitKind::Hero, Team::Dire, 2_300, 2_000),
            unit(
                EntityId {
                    idx: 30,
                    generation: 1,
                },
                UnitKind::Fountain,
                Team::Radiant,
                1_500,
                1_500,
            ),
        ],
        projectiles: Vec::new(),
        players: vec![own, enemy],
        felled_trees: Vec::new(),
        planted_trees: Vec::new(),
        loot: vec![LootView {
            id: LOOT,
            pos: Vec2::from_ints(2_050, 2_000),
            item: ItemId(0),
            charges: Some(1),
        }],
    }
}

fn own_player() -> PlayerView {
    PlayerView {
        slot: SlotId(0),
        team: Team::Radiant,
        hero: SHADOW_FIEND,
        unit: Some(HERO),
        level: 6,
        xp: 0,
        gold: Some(600),
        stash: Some(vec![None; 6]),
        kit: None,
        kills: 0,
        deaths: 0,
        assists: 0,
        last_hits: 0,
        denies: 0,
        respawn_left: 0,
    }
}

fn unit(id: EntityId, kind: UnitKind, team: Team, x: i32, y: i32) -> UnitView {
    fixtures::UnitFixture {
        id,
        kind,
        team,
        pos: Vec2::from_ints(x, y),
        mana: 0,
        attack_damage: 50,
        attack_time: 1000,
        attributes: Attributes::all(20),
        primary: Some(Attribute::Agility),
        hero: (kind == UnitKind::Hero).then_some(HeroId(2)),
        owner: None,
        level: 0,
    }
    .build()
}

pub(super) fn shop_entries(rows: &[(u16, i32, &[u16])]) -> Vec<ShopEntry> {
    assert!(rows.len() <= crate::MAX_SHOP_ITEMS);
    assert!(
        rows.iter()
            .all(|(_, _, parts)| parts.len() <= crate::MAX_SHOP_ITEMS)
    );
    rows.iter()
        .map(|&(id, cost, parts)| ShopEntry {
            id: ItemId(id),
            cost,
            components: parts.iter().copied().map(ItemId).collect(),
        })
        .collect()
}

pub(super) fn ability(aim: Aim, range: i32) -> AbilityView {
    AbilityView {
        id: AbilityId(1),
        level: 1,
        max_level: 4,
        cooldown_left: 0,
        mana_cost: 75,
        range,
        aim,
        passive: false,
        on: false,
        can_level: false,
    }
}

pub(super) fn item(aim: Option<Aim>, range: i32) -> ItemView {
    ItemView {
        id: ItemId(0),
        charges: Some(1),
        cooldown_left: 0,
        mute_left: 0,
        mode: None,
        mana_cost: 0,
        range,
        aim,
        for_sale: false,
    }
}
