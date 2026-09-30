use bota_proto::{AbilitySlot, ItemId, Order, ServerMsg, SlotId, Target, UnitKind, WorldView};

use crate::{
    Arena, ArenaConfig, IssuedOrder, ItemReadiness, OrderPersistence, StateTracker, Teacher,
};

#[test]
fn native_catalog_purchase_prioritizes_recipe_and_retries_rejected_consumable() {
    let (info, mut view) = fixture();
    view.players[0].gold = Some(210);
    equip(&mut view, 0, 9);
    equip(&mut view, 1, 11);
    let tracker = track(&info, &view);
    let mut teacher = Teacher::new();
    let recipe = decision(&mut teacher, &tracker).expect("recipe");
    assert_eq!(recipe.order, Order::Buy { item: ItemId(39) });
    teacher.note_sent(1, recipe, view.tick);
    assert!(!teacher.note_rejected(1));
    assert_eq!(decision(&mut teacher, &tracker), Some(recipe));

    let (info, mut view) = fixture();
    view.players[0].gold = Some(504);
    equip(&mut view, 0, 33);
    let tracker = track(&info, &view);
    let mut teacher = Teacher::new();
    let consumable = IssuedOrder {
        unit: None,
        order: Order::Buy { item: ItemId(7) },
    };
    assert_eq!(decision(&mut teacher, &tracker), Some(consumable));
    teacher.note_sent(9, consumable, view.tick);
    assert_eq!(
        decision(&mut teacher, &tracker).expect("boots").order,
        Order::Buy { item: ItemId(0) }
    );
    assert!(!teacher.note_rejected(8));
    assert!(teacher.note_rejected(9));
    assert_eq!(decision(&mut teacher, &tracker), Some(consumable));
}

#[test]
fn native_courier_collects_stash_then_delivers_loaded_inventory() {
    for loaded in [false, true] {
        let (info, mut view) = fixture();
        equip(&mut view, 0, 7);
        let item = hero_mut(&mut view).items[0].take();
        view.players[0].stash.as_mut().expect("stash")[0] = item;
        let courier = view
            .units
            .iter_mut()
            .find(|unit| unit.kind == UnitKind::Courier && unit.owner == Some(SlotId(0)))
            .expect("native courier");
        if loaded {
            courier.items.fill(item);
        }
        let courier_id = courier.id;
        let issued = decision(&mut Teacher::new(), &track(&info, &view)).expect("courier work");
        assert_eq!(issued.unit, Some(courier_id));
        assert_eq!(
            issued.order,
            Order::Cast {
                slot: AbilitySlot(if loaded { 3 } else { 0 }),
                target: Target::None
            }
        );
    }
}

#[test]
fn native_assembled_build_does_not_rebuy_consumed_components() {
    let (info, mut view) = fixture();
    view.players[0].gold = Some(600);
    equip(&mut view, 0, 29);
    equip(&mut view, 1, 36);
    let mut teacher = Teacher::new();
    for (sequence, id) in [(1, 33), (2, 7)] {
        teacher.note_sent(
            sequence,
            IssuedOrder {
                unit: None,
                order: Order::Buy { item: ItemId(id) },
            },
            view.tick,
        );
    }
    assert!(
        !decision(&mut teacher, &track(&info, &view))
            .is_some_and(|issued| matches!(issued.order, Order::Buy { .. }))
    );
}

fn decision(teacher: &mut Teacher, tracker: &StateTracker) -> Option<IssuedOrder> {
    let (action, space) = teacher
        .decide(tracker, &OrderPersistence::default(), &ItemReadiness::new())
        .expect("decision");
    assert!(space.allows(action));
    space.decode(action).expect("native order")
}

fn fixture() -> (bota_proto::MatchInfo, WorldView) {
    let (_, start) = Arena::new(ArenaConfig {
        seats: 2,
        map: bota_proto::MapId(1),
        seed: 92_009_003,
    })
    .expect("arena");
    let info = start.messages[0]
        .iter()
        .find_map(|message| match message {
            ServerMsg::MatchStart { info } => Some(info.clone()),
            _ => None,
        })
        .expect("match info");
    let mut view = start.messages[0]
        .iter()
        .find_map(|message| match message {
            ServerMsg::Snapshot { view } => Some(view.clone()),
            _ => None,
        })
        .expect("snapshot");
    view.players[0].gold = Some(0);
    for ability in &mut hero_mut(&mut view).abilities {
        ability.can_level = false;
    }
    (info, view)
}

fn track(info: &bota_proto::MatchInfo, view: &WorldView) -> StateTracker {
    let mut tracker = StateTracker::new(SlotId(0), info).expect("tracker");
    tracker.observe_snapshot(view).expect("snapshot");
    tracker
}

fn hero_mut(view: &mut WorldView) -> &mut bota_proto::UnitView {
    let id = view.players[0].unit.expect("hero");
    view.units
        .iter_mut()
        .find(|unit| unit.id == id)
        .expect("own hero")
}

fn equip(view: &mut WorldView, slot: usize, id: u16) {
    let definition = bota_server::game::item_def(ItemId(id)).expect("catalog item");
    hero_mut(view).items[slot] = Some(bota_proto::ItemView {
        id: ItemId(id),
        charges: Some(3),
        cooldown_left: 0,
        mute_left: 0,
        mode: definition.mode,
        mana_cost: definition.mana_cost,
        range: definition.range,
        aim: definition.aim,
        for_sale: false,
        owner: SlotId(0),
    });
}
