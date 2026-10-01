use bota_proto::{
    AbilityId, Angle, DamageKind, EntityId, EventKind, Fixed, MapId, MatchInfo, ProjectileView,
    SlotId, Team, UnitKind, Vec2, WorldView,
};

use super::fixtures;
use crate::{ActionSpace, SHADOW_FIEND, StateTracker};

#[test]
fn metadata_rejects_an_unsupported_map_and_a_zero_tick_rate() {
    for (map, tick_rate, expected) in [
        (
            MapId(3),
            30,
            "unsupported map MapId(3); expected MapId(0), MapId(1), or MapId(2)",
        ),
        (MapId(0), 0, "MatchInfo.tick_rate must be positive"),
    ] {
        let mut info = match_info();
        info.map = map;
        info.tick_rate = tick_rate;
        assert_eq!(
            StateTracker::new(SlotId(0), &info)
                .err()
                .expect("invalid metadata")
                .to_string(),
            expected
        );
    }
}

#[test]
fn invalid_snapshot_is_transactional_and_same_tick_retry_invalidates_only_live_space() {
    type SnapshotChange = fn(&mut WorldView);
    let cases: &[(SnapshotChange, &str)] = &[
        (
            |view| view.viewer = Some(Team::Dire),
            "snapshot viewer Some(Dire) differs from tracker team Radiant",
        ),
        (
            |view| view.tick = 1,
            "snapshot tick 1 does not follow current tick 1",
        ),
        (
            |view| {
                view.players.remove(0);
            },
            "snapshot players has no MatchInfo pick SlotId(0)",
        ),
        (
            |view| view.units[0].pos.x.raw = -1,
            "WorldView.units[0] position raw (-1, 1310720) is outside 0..=33554431",
        ),
        (
            |view| view.units[0].items = vec![None; 10],
            "UnitView.items has 10 entries; limit is 9",
        ),
    ];
    for (change, expected) in cases {
        let mut tracker = new_tracker();
        tracker.observe_snapshot(&world_view(1)).expect("baseline");
        let branch = tracker.clone();
        let space = ActionSpace::from_tracker(&tracker).expect("space");
        let branch_space = ActionSpace::from_tracker(&branch).expect("branch space");
        assert!(!space.matches_tracker(&branch));
        assert!(!branch_space.matches_tracker(&tracker));
        let provenance = tracker.provenance();
        let history = tracker.history();
        let mut invalid = world_view(2);
        change(&mut invalid);
        assert_eq!(
            tracker
                .observe_snapshot(&invalid)
                .expect_err("invalid snapshot")
                .to_string(),
            *expected
        );
        assert!(space.matches_tracker(&tracker));
        assert_eq!(tracker.provenance(), provenance);
        assert_eq!(tracker.history(), history);
        assert_eq!(tracker.current(), branch.current());
        tracker
            .observe_snapshot(&world_view(2))
            .expect("same tick retry");
        assert!(!space.matches_tracker(&tracker));
        assert!(branch_space.matches_tracker(&branch));
    }
}

#[test]
fn snapshot_position_accepts_last_raw_coordinate_but_rejects_the_next_transactionally() {
    let mut view = world_view(1);
    view.units[0].pos = Vec2 {
        x: Fixed { raw: 33_554_431 },
        y: Fixed { raw: 33_554_431 },
    };
    let mut tracker = new_tracker();
    tracker.observe_snapshot(&view).expect("inclusive boundary");
    let baseline = ActionSpace::from_tracker(&tracker).expect("boundary space");
    view.tick = 2;
    view.units[0].pos.x.raw += 1;
    assert_eq!(
        tracker
            .observe_snapshot(&view)
            .expect_err("outside boundary")
            .to_string(),
        "WorldView.units[0] position raw (33554432, 33554431) is outside 0..=33554431"
    );
    assert!(baseline.matches_tracker(&tracker));
}

/// Regression: bota flies requiem lines, hooks and raze marks to full reach from
/// a caster at the map edge; eval seed 1000016 died on a requiem line 6 units
/// west of the map (raw x -388850). Bodies stay strictly on the map.
#[test]
fn projectiles_may_fly_past_the_map_edge_within_the_reach_margin() {
    let margin = crate::PROJECTILE_MAP_MARGIN * Fixed::ONE.raw;
    let mut tracker = new_tracker();
    for (tick, x_raw) in [(1, -388_850), (2, -margin), (3, 33_554_431 + margin)] {
        let mut view = world_view(tick);
        view.projectiles = vec![projectile(Vec2 {
            x: Fixed { raw: x_raw },
            y: Fixed { raw: 1_310_720 },
        })];
        tracker
            .observe_snapshot(&view)
            .expect("projectile within reach");
    }
    let mut view = world_view(4);
    view.projectiles = vec![projectile(Vec2 {
        x: Fixed { raw: -margin - 1 },
        y: Fixed { raw: 1_310_720 },
    })];
    assert_eq!(
        tracker
            .observe_snapshot(&view)
            .expect_err("projectile beyond reach")
            .to_string(),
        "WorldView.projectiles[0] position raw (-134217729, 1310720) is outside -134217728..=167772159"
    );
}

fn projectile(pos: Vec2) -> ProjectileView {
    ProjectileView {
        id: EntityId {
            idx: 900,
            generation: 1,
        },
        pos,
        facing: Angle { brads: 0 },
        team: Team::Radiant,
        ability: Some(AbilityId(16)),
    }
}

#[test]
fn generations_remain_distinct_and_fog_expires_on_game_tick_boundary() {
    let mut first = world_view(1);
    for generation in [1, 2] {
        let mut enemy = first.units[0].clone();
        enemy.id = EntityId {
            idx: 20,
            generation,
        };
        enemy.owner = None;
        enemy.team = Team::Dire;
        first.units.push(enemy);
    }
    let mut tracker = new_tracker();
    tracker.observe_snapshot(&first).expect("two generations");
    let mut next = world_view(6);
    next.units[0].pos = Vec2::from_ints(16, 24);
    next.units[0].hp = 875;
    next.units[0].mana = 360;
    tracker.observe_snapshot(&next).expect("skipped ticks");
    let hero = tracker.entity(entity()).expect("hero");
    let velocity = hero.velocity.expect("velocity");
    assert_eq!(velocity.delta, Vec2::from_ints(6, 4));
    assert_eq!(velocity.elapsed_ticks, 5);
    assert_eq!((hero.hp_delta, hero.mana_delta), (-125, -40));
    tracker.observe_snapshot(&world_view(481)).expect("age 480");
    for generation in [1, 2] {
        assert!(
            !tracker
                .entity(EntityId {
                    idx: 20,
                    generation
                })
                .expect("remembered generation")
                .visible
        );
    }
    tracker.observe_snapshot(&world_view(482)).expect("age 481");
    for generation in [1, 2] {
        assert!(
            tracker
                .entity(EntityId {
                    idx: 20,
                    generation
                })
                .is_none()
        );
    }
}

#[test]
fn events_preserve_cast_ambiguity_and_reject_old_batches_without_mutation() {
    for cast in [false, true] {
        let mut tracker = new_tracker();
        tracker.observe_snapshot(&world_view(1)).expect("snapshot");
        let mut events = vec![EventKind::Damaged {
            source: Some(entity()),
            target: entity(),
            amount: 75,
            kind: DamageKind::Physical,
            crit: false,
        }];
        if cast {
            events.push(EventKind::AbilityCast {
                caster: entity(),
                ability: AbilityId(13),
            });
        }
        events.push(EventKind::Healed {
            source: None,
            target: entity(),
            amount: 20,
            mana: 0,
        });
        tracker.observe_events(1, &events).expect("events");
        let hero = tracker.entity(entity()).expect("hero");
        assert_eq!(hero.last_damage_taken.expect("damage").amount, 75);
        assert_eq!(hero.last_heal_received.expect("heal").amount, 20);
        assert_eq!(hero.last_possible_attack_landed.is_none(), cast);
        let death = EventKind::Died {
            unit: entity(),
            killer: None,
            denied: false,
            gold: 125,
        };
        tracker.observe_events(2, &[death]).expect("death");
        assert!(!tracker.entity(entity()).expect("dead").visible);
        assert_eq!(
            tracker
                .entity(entity())
                .expect("dead")
                .last_death
                .expect("death")
                .gold,
            125
        );
        let before = ActionSpace::from_tracker(&tracker).expect("event space");
        assert_eq!(
            tracker
                .observe_events(2, &[])
                .expect_err("duplicate tick")
                .to_string(),
            "event batch tick 2 must be greater than last event tick 2"
        );
        assert!(before.matches_tracker(&tracker));
        tracker.observe_events(3, &[]).expect("valid retry");
    }
}

#[test]
fn event_stream_is_bounded_independent_and_does_not_invent_unknown_entities() {
    let mut tracker = new_tracker();
    let unknown = EntityId {
        idx: 999,
        generation: 7,
    };
    let death = EventKind::Died {
        unit: unknown,
        killer: None,
        denied: false,
        gold: 0,
    };
    tracker
        .observe_events(10, std::slice::from_ref(&death))
        .expect("events first");
    tracker
        .observe_snapshot(&world_view(20))
        .expect("independent snapshot");
    tracker
        .observe_events(11, &vec![death.clone(); 100])
        .expect("events behind snapshot");
    assert_eq!(tracker.recent_events().len(), 64);
    assert_eq!(tracker.recent_events().back().expect("event").kind, death);
    assert!(tracker.entity(unknown).is_none());
}

fn new_tracker() -> StateTracker {
    StateTracker::new(SlotId(0), &match_info()).expect("tracker")
}

fn match_info() -> MatchInfo {
    fixtures::MatchInfoFixture::new(77, MapId(0), fixtures::two_seat_picks(Team::Radiant))
        .terrain_cells(8)
        .terrain_rle(vec![(64, 0x80)])
        .build()
}

fn world_view(tick: u32) -> WorldView {
    let hero = fixtures::UnitFixture {
        id: entity(),
        kind: UnitKind::Hero,
        team: Team::Radiant,
        pos: Vec2::from_ints(10, 20),
        mana: 400,
        attack_damage: 55,
        attack_time: 1000,
        attributes: bota_proto::Attributes::all(20),
        primary: Some(bota_proto::Attribute::Agility),
        hero: Some(SHADOW_FIEND),
        owner: Some(SlotId(0)),
        level: 3,
    }
    .build();
    let players = [true, false]
        .map(|own| bota_proto::PlayerView {
            slot: SlotId(u8::from(!own)),
            team: if own { Team::Radiant } else { Team::Dire },
            hero: SHADOW_FIEND,
            unit: own.then_some(entity()),
            level: if own { 3 } else { 2 },
            xp: 0,
            gold: own.then_some(600),
            stash: own.then(|| vec![None; 6]),
            kit: None,
            kills: 0,
            deaths: 0,
            assists: 0,
            last_hits: 0,
            denies: 0,
            respawn_left: if own { 0 } else { 10 },
        })
        .to_vec();
    WorldView {
        tick,
        viewer: Some(Team::Radiant),
        units: vec![hero],
        players,
        projectiles: Vec::new(),
        felled_trees: Vec::new(),
        planted_trees: Vec::new(),
        loot: Vec::new(),
    }
}

fn entity() -> EntityId {
    EntityId {
        idx: 1,
        generation: 1,
    }
}
