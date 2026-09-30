#![allow(
    clippy::float_arithmetic,
    reason = "Feature goldens compare bounded normalized values."
)]

use bota_proto::{
    AbilityId, AbilityView, Aim, Angle, Attribute, Attributes, EntityId, EventKind, Fixed, ItemId,
    ItemView, LootView, MapId, MatchInfo, PlayerView, ProjectileView, SlotId, StatusFlags, Team,
    UnitKind, UnitView, Vec2, WorldView,
};

use super::{action, fixtures};
use crate::feature::RaggedFeatureArena;
use crate::{
    ActionSpace, FeatureEncoder, FeatureFrame, GLOBAL_FEATURES, ItemReadiness, LocalPolicyState,
    SHADOW_FIEND, StateTracker, UNIT_FEATURES, ability_feature, global_feature, item_feature,
    loot_feature, projectile_feature, unit_feature,
};

const AXIS: u32 = 128;
const EXTENT: i32 = 8_192;
const HERO: EntityId = entity(10, 1);
const COURIER: EntityId = entity(11, 1);
const ENEMY: EntityId = entity(20, 1);

#[path = "feature_facts.rs"]
mod feature_facts;

#[test]
fn hidden_scoreboard_and_remote_tree_changes_do_not_change_the_policy_frame() {
    let mut view = world_view(Team::Radiant, 10);
    let baseline = encoded_frame(Team::Radiant, view.clone());
    let enemy = &mut view.players[1];
    enemy.xp = 99_999;
    enemy.level = 30;
    enemy.kills = 100;
    enemy.deaths = 100;
    enemy.assists = 100;
    enemy.last_hits = 500;
    enemy.denies = 500;
    enemy.unit = None;
    view.felled_trees.push(1);
    assert_eq!(encoded_frame(Team::Radiant, view), baseline);
}

#[test]
fn observation_trace_keeps_same_tick_events_out_and_expires_fog_without_action_pointers() {
    let mut view = world_view(Team::Radiant, 1);
    let mut tracker = tracker_with_view(Team::Radiant, view.clone());
    let baseline = encode(&tracker, &LocalPolicyState::new(0));
    tracker
        .observe_events(
            1,
            &[EventKind::AbilityCast {
                caster: HERO,
                ability: AbilityId(13),
            }],
        )
        .expect("same-tick event");
    assert_eq!(encode(&tracker, &LocalPolicyState::new(0)), baseline);
    remove_hero_body(&mut view, 10);
    view.units
        .retain(|unit| unit.id != ENEMY && unit.id != COURIER);
    for (tick, present) in [(2, true), (481, true), (482, false)] {
        view.tick = tick;
        tracker.observe_snapshot(&view).expect("fog snapshot");
        let frame = encode(&tracker, &LocalPolicyState::new(0));
        assert_eq!(
            frame.abilities[0][ability_feature::SCOREBOARD_KIT_SOURCE],
            1.0
        );
        assert_eq!(frame.items[0][item_feature::SCOREBOARD_KIT_SOURCE], 1.0);
        assert_eq!(
            frame
                .remembered_units
                .iter()
                .any(|row| row[unit_feature::TOKEN_PRESENT] == 1.0),
            present
        );
        let space = ActionSpace::from_tracker(&tracker).expect("fog action space");
        for id in [HERO, COURIER, ENEMY] {
            assert!(space.entity_index(id).is_none());
        }
    }
}

#[test]
fn stale_encoding_and_foreign_successors_fail_without_changing_observations_or_output() {
    let tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 1));
    let mut encoder = FeatureEncoder::new(&tracker);
    encoder.observe(&tracker).expect("initial observation");
    let expected = encode_with_encoder(&tracker, &mut encoder);
    let space = ActionSpace::from_tracker(&tracker).expect("initial space");
    let mut branch = tracker.clone();
    branch
        .observe_snapshot(&world_view(Team::Radiant, 2))
        .expect("successor");
    let mut output = expected.clone();
    assert_eq!(
        encoder
            .encode(
                &branch,
                &space,
                &ItemReadiness::new(),
                &LocalPolicyState::new(0),
                &mut output
            )
            .expect_err("stale action space")
            .to_string(),
        "feature snapshot tick 2 differs from action-space tick 1"
    );
    assert_eq!(output, expected);
    assert_eq!(
        encoder
            .observe(&branch)
            .expect_err("foreign successor")
            .to_string(),
        "feature observation snapshot tick 2 does not extend its exact predecessor"
    );
    assert_eq!(encode_with_encoder(&tracker, &mut encoder), expected);
}

#[test]
fn coordinate_and_angle_boundaries_are_canonical_on_both_sides() {
    let maximum = ((i64::from(AXIS) * i64::from(crate::TERRAIN_CELL_SIZE)) << Fixed::FRAC_BITS) - 1;
    let maximum = i32::try_from(maximum).expect("fixture extent");
    for (side, position, angle, expected) in [
        (Team::Radiant, 0, 0, 0.0),
        (Team::Dire, maximum, 1 << 15, 0.0),
        (Team::Radiant, maximum, u16::MAX, 1.0),
        (Team::Dire, 0, (1 << 15) - 1, 1.0),
    ] {
        let frame = boundary_frame(side, position, angle);
        assert_eq!(
            [
                frame.units[0][unit_feature::POSITION_X],
                frame.units[0][unit_feature::FACING]
            ],
            [expected; 2]
        );
    }
}

#[test]
fn extreme_resources_and_identifiers_remain_finite_and_use_combat_hulls() {
    for mana in [0, i32::MAX] {
        let mut view = world_view(Team::Radiant, u32::MAX);
        view.players[0].gold = Some(i32::MAX);
        view.players[0].xp = i32::MAX;
        let hero = &mut view.units[0];
        hero.hp = i32::MIN;
        hero.max_hp = i32::MAX;
        hero.mana = mana;
        hero.max_mana = i32::MAX;
        hero.collision = Fixed::from_int(120);
        hero.bound = Fixed::from_int(24);
        hero.abilities[0].id = AbilityId(u16::MAX);
        hero.items[0] = Some(item(ItemId(u16::MAX)));
        view.projectiles[0].ability = Some(AbilityId(u16::MAX));
        view.loot[0].item = ItemId(u16::MAX);
        let frame = encoded_frame(Team::Radiant, view);
        // Nonpositive HP removes the hero from action pointers, not fixed own-body rows.
        assert_eq!(
            [
                frame.global[global_feature::TICK],
                frame.global[global_feature::OWN_GOLD],
                frame.own_units[0][unit_feature::HP_RATIO],
                frame.own_units[0][unit_feature::MANA_PRESENT],
                frame.own_units[0][unit_feature::MANA_RATIO],
                frame.own_units[1][unit_feature::MANA_PRESENT],
                frame.own_units[1][unit_feature::MANA_RATIO],
                frame.own_units[0][unit_feature::RADIUS],
                frame.abilities[0][ability_feature::ID_TOKEN],
                frame.items[0][item_feature::ITEM_TOKEN],
                frame.projectiles[0][projectile_feature::ABILITY_TOKEN],
                frame.loot[0][loot_feature::ITEM_TOKEN],
            ],
            [
                1.0,
                1.0,
                0.0,
                1.0,
                f32::from(mana != 0),
                0.0,
                0.0,
                24.0 / 8192.0,
                65_547.0,
                65_536.0,
                65_547.0,
                65_536.0
            ]
        );
        assert!(
            fixtures::feature_values(&frame, None, None)
                .all(|value| (-1.0..=65_547.0).contains(&value))
        );
    }
}

#[test]
fn representative_frames_match_frozen_goldens_and_resource_presence_boundaries() {
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    for count in [0, 1, 31, 32] {
        let mut view = projectile_capacity_view(Team::Radiant, count);
        let mut tracker = tracker_with_view(Team::Radiant, view.clone());
        let mut encoder = FeatureEncoder::new(&tracker);
        for tick in 1..=2 {
            if tick == 2 {
                view.tick = tick;
                for projectile in &mut view.projectiles {
                    projectile.pos.x += Fixed::from_int(1);
                }
                tracker.observe_snapshot(&view).expect("second observation");
            }
            encoder.observe(&tracker).expect("feature observation");
            let frame = encode_with_encoder(&tracker, &mut encoder);
            assert_current_golden_suffix(&frame);
            let mut arena = RaggedFeatureArena::new(1).expect("golden frame arena");
            let header = arena.push(&frame).expect("store golden frame");
            assert_eq!(arena.expand(&header).expect("expand golden frame"), frame);
            if tick == 1 {
                assert_canonical_golden(&frame, count);
            }
            if count == 1 {
                let expected = if tick == 1 {
                    [0.0; 4]
                } else {
                    [1.0, 0.0, 1.0, 0.0]
                };
                assert_eq!(
                    &frame.units[0][unit_feature::HP_DELTA_PRESENT..=unit_feature::MANA_DELTA],
                    &expected
                );
                let fresh = encode(&tracker, &LocalPolicyState::new(0));
                assert_eq!(
                    &fresh.units[0][unit_feature::HP_DELTA_PRESENT..=unit_feature::MANA_DELTA],
                    &expected
                );
            }
            for value in fixtures::feature_values(&frame, Some(64), Some(69)) {
                digest.update(value.to_bits().to_le_bytes());
            }
            assert!(frame.is_finite());
        }
    }
    let actual: [u8; 32] = digest.finalize().into();
    // Re-captured for action v6, where a raze is legal only with a hostile unit in reach.
    assert_eq!(
        actual,
        [
            56, 182, 3, 3, 115, 209, 128, 113, 75, 144, 36, 147, 191, 143, 156, 42, 198, 144, 23,
            70, 79, 5, 131, 16, 242, 182, 109, 183, 251, 64, 16, 222,
        ]
    );
}

fn assert_canonical_golden(frame: &FeatureFrame, count: u32) {
    let mut canonical = frame.clone();
    canonical.global[global_feature::SIDE_RADIANT..=global_feature::SIDE_DIRE].fill(0.0);
    for side in [Team::Radiant, Team::Dire] {
        let mut remapped = projectile_capacity_view(side, count);
        reverse_entity_ids_and_generations(&mut remapped, 30_000, 77);
        let mut info = match_info(side);
        info.match_id = u64::MAX;
        let mut tracker = StateTracker::new(SlotId(0), &info).expect("remapped tracker");
        tracker
            .observe_snapshot(&remapped)
            .expect("remapped snapshot");
        let mut actual = encode(&tracker, &LocalPolicyState::new(0));
        actual.global[global_feature::SIDE_RADIANT..=global_feature::SIDE_DIRE].fill(0.0);
        assert_eq!(actual, canonical);
    }
}

fn projectile_capacity_view(team: Team, count: u32) -> WorldView {
    assert!(count <= 4_096);
    let mut view = world_view(team, 1);
    let prototype = view.projectiles[0];
    view.projectiles = (0..count)
        .map(|index| {
            let mut projectile = prototype;
            projectile.id = entity(1_000 + index, 1);
            projectile.pos = canonical_position(team, Vec2::from_ints(2_000 + index as i32, 2_000));
            projectile
        })
        .collect();
    assert_eq!(view.projectiles.len(), count as usize);
    view
}

fn assert_current_golden_suffix(frame: &FeatureFrame) {
    // The frozen digest covers the legacy prefix; these facts complete the current frame.
    assert_eq!(&frame.global[64..], &[0.0; GLOBAL_FEATURES - 64]);
    for row in frame
        .units
        .iter()
        .chain(&frame.own_units)
        .chain(&frame.remembered_units)
    {
        assert_eq!(&row[69..71], &[0.0; 2]);
        assert_eq!(&row[73..], &[0.0; UNIT_FEATURES - 73]);
        let facing = if row[unit_feature::TOKEN_PRESENT] == 0.0 {
            [0.0, 0.0]
        } else if row[unit_feature::RELATION_START + 2] == 1.0 {
            [-1.0, 0.0]
        } else if row[unit_feature::KIND_TOKEN] == 1.0 {
            [0.9954076, 0.095727]
        } else {
            [1.0, 0.0]
        };
        for (actual, expected) in row[71..73].iter().zip(facing) {
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "golden facing {actual} != {expected}"
            );
        }
    }
}

#[test]
fn feature_observation_history_has_an_exact_fixed_capacity() {
    let mut tracker = tracker_with_view(Team::Radiant, world_view(Team::Radiant, 1));
    let mut encoder = FeatureEncoder::new(&tracker);
    encoder.observe(&tracker).expect("first observation");
    for tick in 2..=18 {
        tracker
            .observe_snapshot(&world_view(Team::Radiant, tick))
            .expect("new snapshot");
        encoder.observe(&tracker).expect("new observation");
    }
    let latest = encode_with_encoder(&tracker, &mut encoder);
    assert_eq!(
        encoder.rollback(2).expect_err("before horizon").to_string(),
        "feature observation rollback tick 2 is older than earliest supported tick 3"
    );
    assert_eq!(encode_with_encoder(&tracker, &mut encoder), latest);
    encoder.rollback(3).expect("oldest retained observation");
}

#[test]
fn ragged_feature_arena_rejects_corrupt_offsets() {
    let frame = encoded_frame(Team::Radiant, world_view(Team::Radiant, 1));
    let mut arena = RaggedFeatureArena::new(1).expect("arena");
    let mut malformed = arena.push(&frame).expect("frame");
    malformed.corrupt_unit_offset_for_test();
    assert_eq!(
        arena.expand(&malformed).expect_err("invalid offset"),
        "ragged feature range is invalid"
    );
}

fn encoded_frame(team: Team, view: WorldView) -> FeatureFrame {
    let tracker = tracker_with_view(team, view);
    encode(&tracker, &LocalPolicyState::new(0))
}

fn remove_hero_body(view: &mut WorldView, respawn_left: u32) {
    let index = unit_index(view, HERO);
    let hero = view.units.remove(index);
    view.players[0].unit = None;
    view.players[0].respawn_left = respawn_left;
    view.players[0].kit = Some(bota_proto::Kit {
        abilities: hero.abilities,
        items: hero.items,
    });
}

pub(super) fn encode(tracker: &StateTracker, local: &LocalPolicyState) -> FeatureFrame {
    let mut encoder = FeatureEncoder::new(tracker);
    encoder.observe(tracker).expect("feature observation");
    encode_with_local(tracker, &mut encoder, local)
}

fn encode_with_encoder(tracker: &StateTracker, encoder: &mut FeatureEncoder) -> FeatureFrame {
    encode_with_local(tracker, encoder, &LocalPolicyState::new(0))
}

fn encode_with_local(
    tracker: &StateTracker,
    encoder: &mut FeatureEncoder,
    local: &LocalPolicyState,
) -> FeatureFrame {
    let action_space = ActionSpace::from_tracker(tracker).expect("action space");
    let mut frame = FeatureFrame::new();
    encoder
        .encode(
            tracker,
            &action_space,
            &ItemReadiness::new(),
            local,
            &mut frame,
        )
        .expect("feature frame");
    frame
}

pub(super) fn tracker_with_view(team: Team, view: WorldView) -> StateTracker {
    let mut tracker = StateTracker::new(SlotId(0), &match_info(team)).expect("tracker");
    tracker.observe_snapshot(&view).expect("snapshot");
    tracker
}

pub(super) fn match_info(team: Team) -> MatchInfo {
    let trees = canonical_positions(
        team,
        [Vec2::from_ints(2_100, 2_000), Vec2::from_ints(7_000, 7_000)],
    )
    .to_vec();
    let opaque_cells = trees
        .iter()
        .map(|position| {
            (
                u16::try_from(position.x.to_int() / crate::TERRAIN_CELL_SIZE).expect("tree x cell"),
                u16::try_from(position.y.to_int() / crate::TERRAIN_CELL_SIZE).expect("tree y cell"),
            )
        })
        .collect();
    fixtures::MatchInfoFixture::new(99, MapId(0), fixtures::two_seat_picks(team))
        .pregame_ticks(90)
        .trees(trees)
        .terrain_cells(AXIS)
        .terrain_rle(vec![((AXIS * AXIS) as u16, 0x80)])
        .opaque_cells(opaque_cells)
        .shop(action::shop_entries(&[
            (0, 500, &[]),
            (1, 50, &[]),
            (2, 100, &[1]),
        ]))
        .build()
}

pub(super) fn world_view(team: Team, tick: u32) -> WorldView {
    let enemy = opposing(team);
    let positions = canonical_positions(
        team,
        [
            Vec2::from_ints(2_000, 2_000),
            Vec2::from_ints(1_800, 1_800),
            Vec2::from_ints(2_300, 2_000),
            Vec2::from_ints(1_000, 1_000),
            Vec2::from_ints(1_400, 1_400),
            Vec2::from_ints(7_000, 7_000),
            Vec2::from_ints(6_600, 6_600),
        ],
    );
    let mut units = vec![
        hero(team, positions[0], true),
        courier(team, positions[1]),
        hero(enemy, positions[2], false),
        building(entity(30, 1), team, UnitKind::Fountain, positions[3]),
        building(entity(31, 1), team, UnitKind::Tower, positions[4]),
        building(entity(32, 1), enemy, UnitKind::Fountain, positions[5]),
        building(entity(33, 1), enemy, UnitKind::Tower, positions[6]),
    ];
    units.sort_by_key(|unit| unit.id);
    WorldView {
        tick,
        viewer: Some(team),
        units,
        projectiles: vec![ProjectileView {
            id: entity(40, 3),
            pos: canonical_position(team, Vec2::from_ints(2_050, 2_000)),
            facing: canonical_angle(team, Angle { brads: 0 }),
            team,
            ability: Some(AbilityId(13)),
        }],
        players: vec![player(team, true), player(enemy, false)],
        felled_trees: Vec::new(),
        planted_trees: vec![canonical_position(team, Vec2::from_ints(2_200, 2_000))],
        loot: vec![LootView {
            id: entity(50, 9),
            pos: canonical_position(team, Vec2::from_ints(2_000, 2_080)),
            item: ItemId(2),
            charges: Some(1),
        }],
    }
}

fn hero(team: Team, position: Vec2, own: bool) -> UnitView {
    let mut unit = base_unit(
        if own { HERO } else { ENEMY },
        UnitKind::Hero,
        team,
        position,
    );
    unit.max_mana = 400;
    unit.hero = Some(SHADOW_FIEND);
    unit.abilities = (13..=18)
        .map(|id| ability(AbilityId(id), if id <= 15 { Aim::Point } else { Aim::Own }))
        .collect();
    unit.hp = if own { 900 } else { 800 };
    unit.mana = if own { 300 } else { 200 };
    unit.attack_damage = if own { 60 } else { 55 };
    unit.attack_time = if own { 1000 } else { 1067 };
    unit.owner = Some(SlotId(u8::from(!own)));
    unit.level = if own { 5 } else { 4 };
    unit.items = vec![None; 9];
    if own {
        unit.facing = canonical_angle(team, Angle { brads: 1_000 });
        unit.attack_speed = 110;
        unit.armor = Fixed::from_int(3);
        unit.magic_resist = Fixed::from_ratio(1, 4);
        unit.attributes = Attributes::all(20);
        unit.primary = Some(Attribute::Agility);
        unit.items[0] = Some(item(ItemId(1)));
    }
    unit
}

fn courier(team: Team, position: Vec2) -> UnitView {
    let mut unit = base_unit(COURIER, UnitKind::Courier, team, position);
    unit.hp = 250;
    unit.max_hp = 250;
    unit.move_speed = Fixed::from_int(380);
    unit.attack_range = Fixed::ZERO;
    unit.collision = Fixed::from_int(16);
    unit.bound = Fixed::from_int(16);
    unit.vision_radius = Fixed::from_int(500);
    unit.owner = Some(SlotId(0));
    unit.abilities = (8..=12)
        .map(|id| ability(AbilityId(id), Aim::Own))
        .collect();
    unit.items = vec![Some(item(ItemId(2))); 6];
    unit
}

fn building(id: EntityId, team: Team, kind: UnitKind, position: Vec2) -> UnitView {
    let mut unit = base_unit(id, kind, team, position);
    unit.attack_damage = 100;
    unit.attack_range = Fixed::from_int(700);
    unit.attack_time = 1000;
    unit.collision = Fixed::from_int(80);
    unit.bound = Fixed::from_int(80);
    unit
}

fn base_unit(id: EntityId, kind: UnitKind, team: Team, pos: Vec2) -> UnitView {
    let mut unit = fixtures::UnitFixture {
        id,
        kind,
        team,
        pos,
        mana: 0,
        attack_damage: 0,
        attack_time: 0,
        attributes: Attributes::ZERO,
        primary: None,
        hero: None,
        owner: None,
        level: 0,
    }
    .build();
    unit.facing = canonical_angle(team, Angle { brads: 0 });
    unit
}

fn player(team: Team, own: bool) -> PlayerView {
    PlayerView {
        slot: SlotId(u8::from(!own)),
        team,
        hero: SHADOW_FIEND,
        unit: Some(if own { HERO } else { ENEMY }),
        level: if own { 5 } else { 4 },
        xp: if own { 500 } else { 400 },
        gold: own.then_some(1_000),
        stash: own.then(|| vec![Some(item(ItemId(2))); 6]),
        kit: None,
        kills: if own { 2 } else { 1 },
        deaths: if own { 1 } else { 2 },
        assists: if own { 3 } else { 2 },
        last_hits: if own { 20 } else { 15 },
        denies: if own { 4 } else { 2 },
        respawn_left: 0,
    }
}

fn ability(id: AbilityId, aim: Aim) -> AbilityView {
    AbilityView {
        id,
        passive: matches!(id.0, 17 | 18),
        ..action::ability(aim, 600)
    }
}

fn item(id: ItemId) -> ItemView {
    ItemView {
        id,
        charges: Some(2),
        for_sale: true,
        ..action::item(Some(Aim::Own), 0)
    }
}

pub(super) fn reverse_entity_ids_and_generations(
    view: &mut WorldView,
    start: u32,
    generation: u32,
) {
    let old_hero = view.players[0].unit;
    let old_enemy = view.players[1].unit;
    let count = u32::try_from(view.units.len()).expect("bounded units");
    for (offset, unit) in view.units.iter_mut().enumerate() {
        let old = unit.id;
        let offset = u32::try_from(offset).expect("bounded unit index");
        unit.id = entity(start + count - offset, generation + offset % 3);
        if Some(old) == old_hero {
            view.players[0].unit = Some(unit.id);
        }
        if Some(old) == old_enemy {
            view.players[1].unit = Some(unit.id);
        }
    }
    view.units.sort_by_key(|unit| unit.id);
    for (offset, projectile) in view.projectiles.iter_mut().enumerate() {
        projectile.id = entity(start + 1_000 + offset as u32, generation);
    }
    for (offset, loot) in view.loot.iter_mut().enumerate() {
        loot.id = entity(start + 2_000 + offset as u32, generation);
    }
}

fn boundary_frame(team: Team, hero_x_raw: i32, facing: u16) -> FeatureFrame {
    let mut view = world_view(team, 1);
    let hero_index = unit_index(&view, HERO);
    view.units[hero_index].pos.x = Fixed { raw: hero_x_raw };
    view.units[hero_index].facing = Angle { brads: facing };
    encoded_frame(team, view)
}

fn canonical_positions<const COUNT: usize>(team: Team, values: [Vec2; COUNT]) -> [Vec2; COUNT] {
    values.map(|position| canonical_position(team, position))
}

fn canonical_position(team: Team, position: Vec2) -> Vec2 {
    if team == Team::Dire {
        let maximum = (i64::from(EXTENT) << Fixed::FRAC_BITS) - 1;
        Vec2 {
            x: Fixed {
                raw: (maximum - i64::from(position.x.raw)) as i32,
            },
            y: Fixed {
                raw: (maximum - i64::from(position.y.raw)) as i32,
            },
        }
    } else {
        position
    }
}

fn canonical_angle(team: Team, angle: Angle) -> Angle {
    if team == Team::Dire {
        Angle {
            brads: angle.brads.wrapping_sub(1 << 15),
        }
    } else {
        angle
    }
}

fn unit_index(view: &WorldView, id: EntityId) -> usize {
    view.units
        .iter()
        .position(|unit| unit.id == id)
        .expect("fixture unit")
}

const fn entity(idx: u32, generation: u32) -> EntityId {
    EntityId { idx, generation }
}

const fn opposing(team: Team) -> Team {
    match team {
        Team::Radiant => Team::Dire,
        Team::Dire => Team::Radiant,
        Team::Neutral => Team::Neutral,
    }
}
