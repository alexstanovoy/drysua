use super::*;

fn physics(side: usize) -> Physics {
    Physics {
        seed: 10094000 + side as u64,
        side,
        tick: 1201,
        level: 3,
        own_hp: 680,
        enemy_hp: 680,
        mana: 353,
        reach: 450,
        facing: if side == 0 { 0 } else { 32768 },
        creep_hp: None,
        mango: false,
        stash: false,
        gold: 0,
        at_home: false,
        deaths: 0,
    }
}

#[test]
fn corrected_labels_ignore_coverage_and_preserve_exact_replayed_prefix() {
    let prefix = Prefix {
        physics: physics(0),
        actions: vec![cast(0)],
    };
    let mut original = rank::replay(&prefix);
    let (frame, space) = original.prepare();
    let hit_coverage = rank::rank(&prefix);
    let chain_coverage = rank::rank(&prefix);
    assert_eq!(hit_coverage.best, chain_coverage.best);
    assert_eq!(
        hit_coverage
            .estimates
            .iter()
            .map(|row| row.score)
            .collect::<Vec<_>>(),
        chain_coverage
            .estimates
            .iter()
            .map(|row| row.score)
            .collect::<Vec<_>>()
    );
    assert_eq!(original.arena.tick(), 1204);
    assert!(frame.matches_action_space(&space));
    assert!(hit_coverage.best.iter().all(|action| space.allows(*action)));
    eprintln!("stack1 ranking={hit_coverage:?}");
}

#[test]
fn native_learning_spends_exact_points_and_never_learns_low_level_requiem() {
    for level in [3, 4] {
        let game = game(Physics {
            level,
            ..physics(0)
        });
        let hero = game.seats[0].tracker.own_hero().unwrap();
        assert_eq!(hero.abilities[5].id.0, 16);
        assert_eq!(hero.abilities[5].level, 0);
        assert_eq!(hero.abilities[0].level, if level == 3 { 1 } else { 2 });
        assert!(!hero.abilities[5].can_level);
    }
}

#[test]
fn fixture_low_creep_health_survives_initial_settling() {
    let state = game(Physics {
        enemy_hp: 0,
        creep_hp: Some(35),
        ..physics(0)
    });
    let creep = state.seats[0]
        .tracker
        .current()
        .unwrap()
        .units
        .iter()
        .find(|unit| unit.kind == UnitKind::CreepMelee)
        .unwrap();
    assert_eq!(creep.hp, 35);
    assert!(creep.max_hp > creep.hp);
}

#[test]
fn native_ultimate_floor_is_five_rejected_six_allowed_with_unspent_points() {
    for level in [5, 6] {
        let (mut arena, _) = Arena::new(ArenaConfig {
            seats: 2,
            map: MapId(2),
            seed: 10094009,
        })
        .unwrap();
        arena.configure_for_test(|world| {
            let hero = world.seats[0].unit.unwrap();
            world.level.insert(hero, Level(level));
            assert_eq!(world.points_spent(hero), 0);
            assert_eq!(world.learn(hero, 5, &mut Vec::new()), level == 6);
        });
    }
}

#[test]
fn historical_hit_corpus_really_pads_continue_after_a_hit_with_legal_followup_casts() {
    let rows = historical::collect(historical::Spec {
        kind: historical::Kind::Hit,
        side: 0,
        variant: 2,
        seed: 10091704,
    });
    let row = rows.iter().find(|row| row.space.tick() == 1204).unwrap();
    assert_eq!(row.action, StructuredAction::Continue);
    assert!(row.space.allows(cast(1)));
    assert!(row.space.entity_candidates().iter().any(|unit| {
        unit.kind == UnitKind::Hero
            && unit.relation == crate::EntityRelation::Enemy
            && unit
                .unit()
                .effects
                .iter()
                .any(|effect| effect.id.0 == 15 && effect.stacks == Some(1))
    }));
}

#[test]
fn empty_raze_has_lower_full_observed_return_and_pays_mana() {
    let prefix = Prefix {
        physics: Physics {
            enemy_hp: 0,
            ..physics(0)
        },
        actions: vec![],
    };
    let idle = rank::estimate(&prefix, StructuredAction::Continue);
    let wasted = rank::estimate(&prefix, cast(0));
    assert!(
        idle.score > wasted.score + 1e-4,
        "idle={idle:?} wasted={wasted:?}"
    );
    assert!(wasted.metrics.mana >= 75);
}

#[test]
fn active_attack_continue_is_not_worse_than_resending_attack_and_idle_loses_to_attack() {
    let physics = Physics {
        mana: 0,
        creep_hp: Some(550),
        enemy_hp: 0,
        ..physics(0)
    };
    let mut original = game(physics);
    let (_, space) = original.prepare();
    let attack = rank::candidates(&original, &space)
        .into_iter()
        .find(|action| action.kind() == ActionKind::AttackUnit)
        .unwrap();
    let prefix = Prefix {
        physics,
        actions: vec![attack],
    };
    let continuing = rank::estimate(&prefix, StructuredAction::Continue);
    let resent = rank::estimate(&prefix, attack);
    assert!((continuing.score - resent.score).abs() < 1e-9);
    let weak = Physics {
        creep_hp: Some(30),
        ..physics
    };
    let idle = holding_return(weak, StructuredAction::Continue, false);
    let attacked = holding_return(weak, attack_for(weak), false);
    assert!(
        attacked.score > idle.score + 1e-4,
        "idle={idle:?} attacked={attacked:?}"
    );
    assert!(attacked.last_hits > 0);
}

fn attack_for(physics: Physics) -> StructuredAction {
    let mut state = game(physics);
    let (_, space) = state.prepare();
    rank::candidates(&state, &space)
        .into_iter()
        .find(|action| action.kind() == ActionKind::AttackUnit)
        .unwrap()
}

fn holding_return(physics: Physics, first: StructuredAction, attack_after: bool) -> Metrics {
    let mut state = game(physics);
    state.action(first);
    for decision in 1..HORIZON / 3 {
        if state.terminal {
            break;
        }
        let action = if attack_after && decision == 1 {
            let (_, space) = state.prepare();
            rank::candidates(&state, &space)
                .into_iter()
                .find(|action| action.kind() == ActionKind::AttackUnit)
                .unwrap()
        } else {
            StructuredAction::Continue
        };
        state.action(action);
    }
    state.total
}

#[test]
fn physical_last_hit_is_valid_and_waste_before_same_paid_kill_has_lower_return() {
    let scene = Physics {
        enemy_hp: 0,
        creep_hp: Some(35),
        facing: 2048,
        ..physics(0)
    };
    let physical = holding_return(scene, attack_for(scene), false);
    let wasted = holding_return(scene, cast(0), true);
    assert_eq!(physical.last_hits, 1);
    assert_eq!(wasted.last_hits, 1);
    assert_eq!(physical.gold, wasted.gold);
    assert!(
        physical.score > wasted.score + 1e-4,
        "physical={physical:?} wasted={wasted:?}"
    );
    assert_eq!(physical.mana, 0);
    assert!(wasted.mana >= 75);
}

#[test]
fn stack_one_continuation_is_selected_only_from_higher_observed_return() {
    let prefix = Prefix {
        physics: physics(0),
        actions: vec![cast(0)],
    };
    let ranking = rank::rank(&prefix);
    let idle = ranking
        .estimates
        .iter()
        .find(|row| row.action == StructuredAction::Continue)
        .unwrap();
    let best = ranking
        .estimates
        .iter()
        .find(|row| row.action == ranking.best[0])
        .unwrap();
    assert!(best.score > idle.score + 1e-4);
    assert!(best.metrics.hit_slots.count_ones() >= 3);
    assert!(
        ranking
            .best
            .iter()
            .any(|action| action.kind() == ActionKind::Cast)
    );
}

#[test]
fn only_native_match_over_supplies_terminal_reward_not_horizon_or_demonstration_end() {
    let mut empty = game(Physics {
        enemy_hp: 0,
        ..physics(0)
    });
    for _ in 0..HORIZON / 3 {
        empty.action(StructuredAction::Continue);
    }
    assert!(!empty.terminal);
    assert!(empty.total.score.abs() < 0.4);
    let mut lethal = game(Physics {
        enemy_hp: 30,
        deaths: 1,
        ..physics(0)
    });
    lethal.action(cast(1));
    assert!(lethal.terminal);
    assert!(lethal.total.score > 0.9);
}

#[test]
fn checked_action_set_rejects_duplicates_and_preserves_full_autoregressive_training() {
    let mut game = game(physics(0));
    let (frame, space) = game.prepare();
    let error = CheckedActionSet::new(frame.clone(), &space, &[cast(0), cast(0)]).unwrap_err();
    assert_eq!(
        error,
        crate::ModelError::InvalidModelState("duplicate advantage action")
    );
    let set = CheckedActionSet::new(frame, &space, &[cast(0), cast(1), cast(2)]).unwrap();
    let model = PolicyModel::fresh_on(10094999, PolicyDevice::Cpu).unwrap();
    let mut optimizer = model.claim_optimizer(AdamConfig::default()).unwrap();
    let update = model
        .train_checked_action_sets(&[&set], &mut optimizer)
        .unwrap();
    assert_eq!(update.optimizer_step, 1);
    assert!(update.average_loss.is_finite());
}

#[test]
fn common_teacher_does_not_realize_mango_mana_value_so_ambiguous_rows_are_skipped() {
    let prefix = Prefix {
        physics: Physics {
            seed: 10094096,
            tick: 901,
            own_hp: 80,
            enemy_hp: 60,
            mana: 0,
            mango: true,
            deaths: 1,
            ..physics(0)
        },
        actions: vec![],
    };
    let immediate = rank::estimate(&prefix, use_mango());
    let delayed = rank::estimate(&prefix, StructuredAction::Continue);
    assert!(immediate.metrics.mango > 0);
    assert_eq!(delayed.metrics.mango, 0);
    assert_eq!(immediate.score, 0.0);
    assert_eq!(delayed.score, 0.0);
    let ranking = rank::rank(&prefix);
    assert_eq!(ranking.separation, 0.0);
}
