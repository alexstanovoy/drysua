use bota_proto::{
    DamageKind, EntityId, EventKind, MapId, Order, ServerMsg, SlotId, Target, Team, UnitKind, Vec2,
    WorldView,
};

use crate::{Arena, ArenaConfig, LaningObserver, LaningProxy, LaningScript, Request};

#[path = "laning_visibility.rs"]
mod visibility;

const LANING_TICKS: u32 = 6_300;

#[test]
fn laning_damage_counts_each_hero_hit_once_and_excludes_npcs_healing_and_friendly_fire() {
    let view = fixture();
    let own = view.players[0].unit.unwrap();
    let enemy = view.players[1].unit.unwrap();
    let npc = EntityId {
        idx: 9999,
        generation: 1,
    };
    let mut observer = LaningObserver::new(SlotId(0));
    observer.observe(&ServerMsg::Snapshot { view }).unwrap();
    let message = ServerMsg::Events {
        tick: 1,
        events: vec![
            damage(own, enemy, 20, DamageKind::Physical),
            damage(own, enemy, 20, DamageKind::Physical),
            damage(own, enemy, 30, DamageKind::Magical),
            damage(own, npc, 1000, DamageKind::Magical),
            damage(npc, enemy, 1000, DamageKind::Physical),
            damage(own, own, 1000, DamageKind::Physical),
            damage(enemy, own, 9, DamageKind::Physical),
            damage(own, enemy, 0, DamageKind::Physical),
            EventKind::Healed {
                source: Some(own),
                target: enemy,
                amount: 1000,
                mana: 0,
            },
        ],
    };

    observer.observe(&message).unwrap();

    assert_eq!(observer.metrics.hero_damage, 70);
    assert_eq!(observer.metrics.hero_physical_hits, 2);
    assert_eq!(observer.metrics.hero_magical_hits, 1);
    assert_eq!(observer.metrics.hero_magical_damage, 30);
    assert_eq!(observer.metrics.hero_damage_taken, 9);
    let before = observer.metrics.clone();
    assert_eq!(
        observer.observe(&message).unwrap_err().to_string(),
        "laning Events must complete exactly one matching Snapshot"
    );
    assert_eq!(observer.metrics, before);
}

#[test]
fn laning_damage_filters_bounded_seeded_event_fuzz() {
    let view = fixture();
    let own = view.players[0].unit.unwrap();
    let enemy = view.players[1].unit.unwrap();
    let npc = EntityId {
        idx: 9999,
        generation: 1,
    };
    let mut random = 0x1a91_0924_0101_u64;
    for case in 0..128 {
        let mut observer = LaningObserver::new(SlotId(0));
        let mut events = Vec::with_capacity(16);
        let mut expected = 0;
        for _ in 0..16 {
            random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
            let source = [own, enemy, npc][((random >> 32) % 3) as usize];
            let target = [own, enemy, npc][((random >> 40) % 3) as usize];
            let amount = ((random >> 48) % 301) as i32 - 1;
            let kind = [DamageKind::Physical, DamageKind::Magical, DamageKind::Pure]
                [((random >> 56) % 3) as usize];
            if source == own && target == enemy && amount > 0 {
                expected += amount as u64;
            }
            events.push(damage(source, target, amount, kind));
        }
        observe_tick(&mut observer, view.clone(), events);
        assert_eq!(
            observer.metrics.hero_damage, expected,
            "seed=0x1a9109240101 case={case}"
        );
    }
}

#[test]
fn laning_lethal_damage_retains_public_hero_identity_but_not_reused_entity_generation() {
    let mut view = fixture();
    let own = view.players[0].unit.unwrap();
    let enemy = view.players[1].unit.unwrap();
    let mut observer = LaningObserver::new(SlotId(0));
    observe_tick(&mut observer, view.clone(), vec![]);
    view.tick = 2;
    view.players[1].unit = None;
    view.units.retain(|unit| unit.id != enemy);

    observe_tick(
        &mut observer,
        view,
        vec![
            damage(own, enemy, 51, DamageKind::Magical),
            damage(
                own,
                EntityId {
                    generation: enemy.generation + 1,
                    ..enemy
                },
                900,
                DamageKind::Magical,
            ),
        ],
    );

    assert_eq!(observer.metrics.hero_damage, 51);
}

#[test]
fn laning_casts_require_cooldown_reset_not_learn_events_or_repeated_event_names() {
    let mut view = fixture();
    let own = view.players[0].unit.unwrap();
    let ability = view
        .units
        .iter()
        .find(|unit| unit.id == own)
        .unwrap()
        .abilities[0]
        .id;
    let cast = EventKind::AbilityCast {
        caster: own,
        ability,
    };
    let mut observer = LaningObserver::new(SlotId(0));
    observe_tick(&mut observer, view.clone(), vec![]);
    view.tick = 2;
    view.units
        .iter_mut()
        .find(|unit| unit.id == own)
        .unwrap()
        .abilities[0]
        .level = 1;
    observe_tick(&mut observer, view.clone(), vec![cast.clone()]);
    assert_eq!(observer.metrics.confirmed_casts, 0);
    view.tick = 3;
    view.units
        .iter_mut()
        .find(|unit| unit.id == own)
        .unwrap()
        .abilities[0]
        .cooldown_left = 300;

    observe_tick(
        &mut observer,
        view.clone(),
        vec![cast.clone(), cast.clone()],
    );

    assert_eq!(observer.metrics.confirmed_casts, 1);
    view.tick = 4;
    let held = &mut view
        .units
        .iter_mut()
        .find(|unit| unit.id == own)
        .unwrap()
        .abilities[0];
    held.cooldown_left = 299;
    held.level = 2;
    observe_tick(&mut observer, view, vec![cast]);
    assert_eq!(observer.metrics.confirmed_casts, 1);
}

#[test]
fn laning_phase_counts_alive_xp_range_boundary_and_excludes_recently_damaged_retreats() {
    let mut view = fixture();
    let own = view.players[0].unit.unwrap();
    let mut creep = view
        .units
        .iter()
        .find(|unit| unit.id == own)
        .unwrap()
        .clone();
    creep.id = EntityId {
        idx: 9999,
        generation: 1,
    };
    creep.owner = None;
    creep.kind = UnitKind::CreepMelee;
    creep.team = Team::Dire;
    let hero_position = Vec2::from_ints(1000, 1000);
    creep.pos = Vec2::from_ints(2500, 1000);
    view.units
        .iter_mut()
        .find(|unit| unit.id == own)
        .unwrap()
        .pos = hero_position;
    view.units.push(creep);
    view.tick = 2999;
    let mut observer = LaningObserver::new(SlotId(0));
    observe_tick(&mut observer, view.clone(), vec![]);
    view.tick = 3000;
    observe_tick(&mut observer, view.clone(), vec![]);
    assert_eq!(observer.metrics.phase_alive_ticks, 1);
    assert_eq!(observer.metrics.phase_xp_ticks, 1);
    assert_eq!(observer.metrics.healthy_behind_tower_ticks, 1);
    view.tick = 6000;
    view.units.last_mut().unwrap().pos = Vec2::from_ints(2501, 1000);
    observe_tick(
        &mut observer,
        view.clone(),
        vec![damage(
            view.players[1].unit.unwrap(),
            own,
            1,
            DamageKind::Physical,
        )],
    );
    assert_eq!(observer.metrics.phase_alive_ticks, 2);
    assert_eq!(observer.metrics.phase_xp_ticks, 1);
    assert_eq!(observer.metrics.healthy_behind_tower_ticks, 1);
    view.tick = 6001;
    observe_tick(&mut observer, view, vec![]);
    assert_eq!(observer.metrics.phase_ticks, 2);
}

#[test]
fn laning_rejects_spectator_views_and_scripts_never_target_hidden_scoreboard_heroes() {
    let mut view = fixture();
    let mut script = LaningScript::new(SlotId(0), LaningProxy::RightClick, 900);
    let enemy = view.players[1].unit.unwrap();
    assert!(!view.units.iter().any(|unit| unit.id == enemy));
    for unit in &mut view.units {
        for ability in &mut unit.abilities {
            ability.can_level = false;
        }
    }
    assert!(
        !matches!(script.decide(&view).unwrap(), Some(Order::Attack { target: Target::Unit(id) }) if id == enemy)
    );
    view.viewer = None;
    assert_eq!(
        script.decide(&view).unwrap_err().to_string(),
        "laning requires the assigned seat's fogged two-player view"
    );
    let mut observer = LaningObserver::new(SlotId(0));
    assert_eq!(
        observer
            .observe(&ServerMsg::Snapshot { view })
            .unwrap_err()
            .to_string(),
        "laning requires the assigned seat's fogged two-player view"
    );
}

#[test]
fn laning_rejects_unfogged_scoreboard_data_even_when_viewer_is_set() {
    let mut view = fixture();
    view.players[1].gold = Some(5000);
    let mut observer = LaningObserver::new(SlotId(0));
    assert_eq!(
        observer
            .observe(&ServerMsg::Snapshot { view })
            .unwrap_err()
            .to_string(),
        "laning requires the assigned seat's fogged two-player view"
    );
    let source = include_str!("../laning_evaluation.rs");
    for forbidden in [
        "bota_server",
        "Teacher",
        "StateTracker",
        "GlobalSummary",
        "world.view",
    ] {
        assert!(
            !source.contains(forbidden),
            "evaluation consumed forbidden input: {forbidden}"
        );
    }
}

#[test]
fn laning_dead_heroes_do_not_count_as_alive_or_occupying_xp_range() {
    let mut view = fixture();
    view.tick = 3000;
    view.players[0].unit = None;
    let mut observer = LaningObserver::new(SlotId(0));
    observe_tick(&mut observer, view, vec![]);
    assert_eq!(observer.metrics.phase_ticks, 1);
    assert_eq!(observer.metrics.phase_alive_ticks, 0);
    assert_eq!(observer.metrics.phase_xp_ticks, 0);
    assert_eq!(observer.metrics.healthy_behind_tower_ticks, 0);
}

#[test]
fn laning_proxies_recover_at_low_health_instead_of_suiciding_before_the_measured_phase() {
    let mut view = fixture();
    let own = view.players[0].unit.unwrap();
    view.units
        .iter_mut()
        .find(|unit| unit.id == own)
        .unwrap()
        .hp = 100;
    let fountain = view
        .units
        .iter()
        .find(|unit| unit.kind == UnitKind::Fountain && unit.team == Team::Radiant)
        .unwrap()
        .pos;
    for proxy in [
        LaningProxy::RightClick,
        LaningProxy::RazeFarm,
        LaningProxy::LanePush,
    ] {
        let mut script = LaningScript::new(SlotId(0), proxy, 900);
        assert_eq!(
            script.decide(&view).unwrap(),
            Some(Order::Move {
                target: Target::Pos(fountain)
            })
        );
    }
}

#[test]
fn laning_scripts_exercise_real_attacks_and_razes_on_paired_seeds_without_win_gates() {
    for proxy in [
        LaningProxy::RightClick,
        LaningProxy::RazeFarm,
        LaningProxy::LanePush,
    ] {
        let mut hero_hits = 0;
        let mut casts = 0;
        let mut phase_ticks = 0;
        for seed in [9_240_101, 9_240_102] {
            for seat in 0..2 {
                let (observers, pregame) = scripted_match(seed, seat, proxy);
                assert!(
                    observers[seat].metrics.unit_physical_hits > 0,
                    "{proxy:?} seed={seed} seat={seat}: {:?}",
                    observers[seat].metrics
                );
                assert!(
                    observers[seat]
                        .metrics
                        .first_center_tick
                        .is_some_and(|tick| tick < pregame)
                );
                assert_eq!(
                    observers[seat].metrics.rejections, 0,
                    "{proxy:?} seed={seed} seat={seat}"
                );
                hero_hits += observers[seat].metrics.hero_physical_hits;
                casts += observers[seat].metrics.confirmed_casts;
                phase_ticks += observers[seat].metrics.phase_ticks;
            }
        }
        if proxy != LaningProxy::LanePush {
            assert!(hero_hits > 0, "{proxy:?}");
        }
        assert_eq!(casts > 0, proxy == LaningProxy::RazeFarm);
        assert!(
            phase_ticks > 0,
            "{proxy:?} must reach the observation phase in at least one paired game"
        );
    }
}

fn scripted_match(seed: u64, seat: usize, proxy: LaningProxy) -> ([LaningObserver; 2], u32) {
    let (mut arena, start) = Arena::new(ArenaConfig {
        seats: 2,
        map: MapId(1),
        seed,
    })
    .unwrap();
    let pregame = start.messages[0]
        .iter()
        .find_map(|message| match message {
            ServerMsg::MatchStart { info } => Some(info.pregame_ticks),
            _ => None,
        })
        .unwrap();
    let mut scripts = std::array::from_fn::<_, 2, _>(|index| {
        LaningScript::new(
            SlotId(index as u8),
            if index == seat {
                proxy
            } else {
                LaningProxy::RightClick
            },
            pregame,
        )
    });
    let mut observers = [
        LaningObserver::new(SlotId(0)),
        LaningObserver::new(SlotId(1)),
    ];
    let mut messages = start.messages;
    let mut visibility = visibility::VisibilityAudit::default();
    for tick in 1..=LANING_TICKS {
        visibility.observe(&messages);
        let requests: [_; 2] = std::array::from_fn(|index| {
            script_request(
                &mut scripts[index],
                &mut observers[index],
                &messages[index],
                tick,
            )
        });
        // Damage delivery follows impact visibility, not symmetric attacker/victim accounting.
        visibility.assert_observers(&observers);
        if tick == LANING_TICKS || observers[0].metrics.winner.is_some() {
            break;
        }
        messages = arena.step(&requests).unwrap().messages;
    }
    visibility.report(seed, seat, proxy);
    (observers, pregame)
}

fn script_request(
    script: &mut LaningScript,
    observer: &mut LaningObserver,
    messages: &[ServerMsg],
    tick: u32,
) -> Option<Request> {
    for message in messages {
        observer.observe(message).unwrap();
    }
    let view = messages
        .iter()
        .find_map(|message| match message {
            ServerMsg::Snapshot { view } => Some(view),
            _ => None,
        })
        .unwrap();
    let order = script.decide(view).unwrap();
    if !(tick - 1).is_multiple_of(3) {
        assert_eq!(order, None);
    }
    order.map(|order| Request {
        seq: tick,
        unit: None,
        order,
    })
}

fn fixture() -> WorldView {
    let (_, start) = Arena::new(ArenaConfig {
        seats: 2,
        map: MapId(1),
        seed: 9_240_101,
    })
    .unwrap();
    start.messages[0]
        .iter()
        .find_map(|message| match message {
            ServerMsg::Snapshot { view } => Some(view.clone()),
            _ => None,
        })
        .unwrap()
}

fn damage(source: EntityId, target: EntityId, amount: i32, kind: DamageKind) -> EventKind {
    EventKind::Damaged {
        source: Some(source),
        target,
        amount,
        kind,
        crit: false,
    }
}

fn observe_tick(observer: &mut LaningObserver, view: WorldView, events: Vec<EventKind>) {
    let tick = view.tick;
    observer.observe(&ServerMsg::Snapshot { view }).unwrap();
    observer
        .observe(&ServerMsg::Events { tick, events })
        .unwrap();
}
