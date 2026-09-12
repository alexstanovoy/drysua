use super::*;
#[path = "map2_recovery_003.rs"]
mod correction;
use sha2::{Digest, Sha256};

struct Row {
    set: CheckedActionSet,
    identity: String,
}

fn trip_data(training: bool, report: &mut String) -> Vec<Row> {
    let variants = if training { &[0, 2][..] } else { &[1, 3][..] };
    let mut rows = Vec::new();
    for setup in trip::setups(training, variants) {
        let mut state = trip::create(setup);
        let start = state.arena.tick();
        let (_, space) = state.prepare();
        let (_, goal) = trip::fountain(&space);
        let mut trace = trip::Trace::default();
        trace.observe(&state, start, goal);
        let mut trajectory = Vec::new();
        let mut ledger = Vec::new();
        let mut last_send = 0;
        for decision in 0..600 {
            if state.terminal {
                break;
            }
            let (frame, space) = state.prepare();
            let action = trip::reference(&state, &space);
            let close = state.seats[setup.side]
                .tracker
                .own_hero()
                .is_some_and(|hero| trip::distance(hero.pos, goal) <= 64.0);
            let sequence = state.seats[setup.side].sequence;
            trip::act(
                &mut state,
                action,
                setup.threat != trip::Threat::None,
                |state| trace.observe(state, start, goal),
            );
            let sent = sequence != state.seats[setup.side].sequence;
            if sent {
                last_send = decision;
            }
            if decision == 0 || sent || decision <= last_send + 3 || decision % 10 == 0 || close {
                trajectory.push(Row {
                    set: CheckedActionSet::new(frame, &space, &[action]).unwrap(),
                    identity: format!(
                        "validated_state_only_reference:{setup:?}:tick{}",
                        space.tick()
                    ),
                });
            }
            ledger.push(action);
            if trip::valid_demonstration(setup, &trace) {
                break;
            }
        }
        let valid = trip::valid_demonstration(setup, &trace);
        writeln!(report, "trip_setup={setup:?} validated={valid} rows={} trace={trace:?} metrics={:?} ledger={ledger:?}", trajectory.len(), state.total).unwrap();
        if valid {
            rows.extend(trajectory);
        }
    }
    assert!(rows.len() <= if training { 2600 } else { 1500 });
    assert!(!rows.is_empty());
    rows
}

fn combat_setups(training: bool) -> Vec<(&'static str, Physics)> {
    super::super::fit::cases(training)
        .into_iter()
        .filter(|(coverage, _)| *coverage != "recover")
        .map(|(coverage, mut physics)| {
            physics.seed += 4400;
            (coverage, physics)
        })
        .collect()
}

fn combat_data(frozen: &PolicyModel, report: &mut String) -> Vec<Row> {
    let mut rows = Vec::new();
    for (coverage, physics) in combat_setups(true) {
        let mut prefix = Prefix {
            physics,
            actions: vec![],
        };
        for decision in 0..60 {
            if rank::replay(&prefix).terminal {
                break;
            }
            let retain = [0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50].contains(&decision);
            let tick = physics.tick + decision * 3;
            let (row, action) = bootstrap_visit(&mut prefix, frozen, retain);
            if let Some((set, _)) = row {
                rows.push(Row {
                    set,
                    identity: format!("neural_counterfactual:{coverage}:{physics:?}:tick{tick}"),
                });
            }
            writeln!(report, "combat_visit seed={} side={} tick={tick} queried={retain} bootstrap={action:?} ledger_len={}", physics.seed, physics.side, prefix.actions.len()).unwrap();
        }
        writeln!(
            report,
            "combat_setup={physics:?} ledger={:?}",
            prefix.actions
        )
        .unwrap();
    }
    assert!(rows.len() <= 600);
    assert!(!rows.is_empty());
    rows
}

fn ordinary(seed: u64, side: usize) -> Game {
    let (arena, start) = Arena::new(ArenaConfig {
        seats: 2,
        map: MapId(2),
        seed,
    })
    .unwrap();
    Game {
        arena,
        seats: setup_seats(start).unwrap(),
        side,
        total: Metrics::default(),
        terminal: false,
    }
}

fn teacher_sample(
    state: &mut Game,
) -> (FeatureFrame, ActionSpace, StructuredAction, Option<Request>) {
    let seat = &mut state.seats[state.side];
    let (frame, _) = prepare_neural_observer_sample(seat).unwrap();
    let (action, space) = seat
        .teacher
        .decide(&seat.tracker, &seat.persistence, &seat.readiness)
        .unwrap();
    seat.local
        .note_decision(space.tick(), action.kind())
        .unwrap();
    let request = issue_request(
        seat,
        space.decode(action).unwrap(),
        &space,
        action.kind(),
        true,
    )
    .unwrap();
    (frame, space, action, request)
}

fn rehearsal(report: &mut String) -> Vec<Row> {
    let mut rows = Vec::new();
    for seed in [10098980, 10098981] {
        for side in 0..2 {
            rehearsal_game(seed, side, &mut rows, report);
        }
    }
    assert!(rows.len() <= 1024);
    rows
}

fn rehearsal_game(seed: u64, side: usize, rows: &mut Vec<Row>, report: &mut String) {
    let mut state = ordinary(seed, side);
    let mut retained = 0;
    let mut levels = [0u32; 11];
    for decision in 0..1500 {
        if state.terminal {
            break;
        }
        let before = state.seats[side].tracker.own_hero().cloned();
        let metrics = state.total.clone();
        let (frame, space, action, request) = teacher_sample(&mut state);
        let active = state.seats[side].local.active_order();
        let other = teacher_request(&mut state.seats[1 - side]).unwrap();
        for tick in 0..3 {
            if state.terminal {
                break;
            }
            state.tick(
                if tick == 0 { request } else { None },
                if tick == 0 { other } else { None },
            );
        }
        let after = state.seats[side].tracker.own_hero();
        let progressing = before.as_ref().zip(after).is_some_and(|(before, after)| {
            before.id == after.id && trip::distance(before.pos, after.pos) > 0.5
        });
        let keep = match action {
            StructuredAction::Learn { slot } => {
                before.as_ref().zip(after).is_some_and(|(before, after)| {
                    after.abilities[slot.0 as usize].level > before.abilities[slot.0 as usize].level
                })
            }
            StructuredAction::Buy { .. } => request.is_some(),
            StructuredAction::Cast {
                unit: ControlledUnit::Hero,
                ..
            } => {
                state.total.casts > metrics.casts
                    && (state.total.damage > metrics.damage || state.total.gold > metrics.gold)
            }
            StructuredAction::AttackUnit { .. } => {
                request.is_some() && (progressing || state.total.damage > metrics.damage)
            }
            StructuredAction::MovePoint { .. } | StructuredAction::AttackMovePoint { .. } => {
                request.is_some() && progressing
            }
            StructuredAction::Continue => active.is_some() && progressing && decision % 20 == 0,
            _ => false,
        };
        if keep && retained < 256 {
            if let Some(hero) = before {
                levels[hero.level as usize] += 1;
            }
            rows.push(Row {
                set: CheckedActionSet::new(frame, &space, &[action]).unwrap(),
                identity: format!(
                    "unchanged_teacher_actual_4500:{seed}:side{side}:tick{}",
                    space.tick()
                ),
            });
            retained += 1;
        }
    }
    writeln!(report, "rehearsal seed={seed} side={side} tick={} terminal={} retained={retained} levels={levels:?}", state.arena.tick(), state.terminal).unwrap();
}

#[test]
#[ignore = "One guarded M17 recovery curriculum, frozen neural counterfactuals, fresh optimizer."]
fn collect_freeze_and_fit() {
    assert!(output().join("CALIBRATION.txt").exists());
    assert!(!output().join("weights").exists());
    let started = Instant::now();
    let frozen = parent();
    let parent_identity = frozen.policy_identity().unwrap();
    let mut data_report = String::new();
    let trips = trip_data(true, &mut data_report);
    let combat = combat_data(&frozen, &mut data_report);
    let ordinary = rehearsal(&mut data_report);
    let total = trips.len() + combat.len() + ordinary.len();
    assert!(total <= 4096);
    let hashes = [data_hash(&trips), data_hash(&combat), data_hash(&ordinary)];
    writeln!(data_report, "trip_rows={} combat_rows={} ordinary_rows={} total={total}\ndata_hashes={hashes:?}\nsource_hashes={:?}\nparent_sha={PARENT_SHA}\nmodel=17 feature=15 action=5 ppo=30 rules=25\ncollection_seconds={:.3}\nfrozen_before_optimizer=true", trips.len(), combat.len(), ordinary.len(), source_hashes(), started.elapsed().as_secs_f64()).unwrap();
    write_new(&output().join("DATASET.txt"), &data_report);
    let model = parent();
    let mut report = String::new();
    let steps = fit(&model, [&trips, &combat, &ordinary], started, &mut report);
    assert_eq!(
        hashes,
        [data_hash(&trips), data_hash(&combat), data_hash(&ordinary)]
    );
    assert_eq!(frozen.policy_identity().unwrap(), parent_identity);
    writeln!(
        report,
        "steps={steps} data_hashes={hashes:?} wall_seconds={:.3}",
        started.elapsed().as_secs_f64()
    )
    .unwrap();
    write_new(&output().join("FIT.txt"), &report);
    assert_eq!(steps, 512, "bounded fit incomplete; no automatic retry");
    let target = output().join("weights");
    std::fs::create_dir(&target).unwrap();
    TrainingArtifact::save_runtime_weights(&model, &target).unwrap();
    let hash = file_hash(&target.join("drysua.weights.safetensors"));
    write_new(
        &target.join("MANIFEST.txt"),
        &format!(
            "diagnostic_only=true\nqualified=false\nmodel=17 feature=15 action=5 ppo=30 rules=25\nweights_sha256={hash}\nparent_sha256={PARENT_SHA}\noptimizer=fresh_Adam_lr1e-4_beta.9_.999_eps1e-8_clip.5\nsteps=512 batch=16 trip=8 combat=4 rehearsal=4 seed=10098999\ndata_hashes={hashes:?}\ndataset_frozen_sha={}\nsource_hashes={:?}\n",
            file_hash(&output().join("DATASET.txt")),
            source_hashes()
        ),
    );
    eprintln!(
        "rows={total} steps={steps} seconds={:.3} weights_sha={hash}",
        started.elapsed().as_secs_f64()
    );
}

fn fit(model: &PolicyModel, groups: [&Vec<Row>; 3], started: Instant, report: &mut String) -> u64 {
    let mut optimizer = model
        .claim_optimizer(AdamConfig {
            learning_rate: 0.0001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            gradient_clip: 0.5,
        })
        .unwrap();
    let mut random = PpoRng::new(10098999);
    let fit_started = Instant::now();
    for step in 1..=512 {
        if started.elapsed() >= Duration::from_secs(265) {
            break;
        }
        let mut batch = Vec::with_capacity(16);
        for (group, count) in groups.iter().zip([8, 4, 4]) {
            assert!(!group.is_empty());
            for _ in 0..count {
                batch.push(&group[random.below(group.len() as u64).unwrap() as usize].set);
            }
        }
        let updated = model
            .train_checked_action_sets(&batch, &mut optimizer)
            .unwrap();
        if step == 1 || step % 64 == 0 {
            writeln!(
                report,
                "step={step} loss={} fit_seconds={:.3}",
                updated.average_loss,
                fit_started.elapsed().as_secs_f64()
            )
            .unwrap();
            eprintln!(
                "step={step} loss={} seconds={:.3}",
                updated.average_loss,
                fit_started.elapsed().as_secs_f64()
            );
        }
    }
    optimizer.step()
}

fn data_hash(rows: &[Row]) -> String {
    let mut hash = Sha256::new();
    for row in rows {
        hash.update(format!("{}:{:?}", row.identity, row.set.actions()));
        let frame = row.set.frame();
        for value in frame
            .global
            .iter()
            .chain(frame.history.iter().flatten())
            .chain(frame.policy_history.iter().flatten())
            .chain(frame.units.iter().flatten())
            .chain(frame.own_units.iter().flatten())
            .chain(frame.remembered_units.iter().flatten())
            .chain(frame.points.iter().flatten())
            .chain(frame.abilities.iter().flatten())
            .chain(frame.items.iter().flatten())
            .chain(frame.projectiles.iter().flatten())
            .chain(frame.loot.iter().flatten())
            .chain(frame.map.iter())
        {
            hash.update(value.to_bits().to_le_bytes());
        }
    }
    hash.finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn source_hashes() -> Vec<(&'static str, String)> {
    [
        "src/tests/map2_recovery_002.rs",
        "src/tests/map2_recovery_002_trip.rs",
        "src/tests/map2_recovery_002_fit.rs",
        "src/tests/map2_advantage.rs",
        "src/tests/map2_advantage_rank.rs",
        "src/model.rs",
        "src/model_advantage_fit.rs",
        "src/model_transfer_fit.rs",
    ]
    .into_iter()
    .map(|name| {
        (
            name,
            file_hash(&Path::new(env!("CARGO_MANIFEST_DIR")).join(name)),
        )
    })
    .collect()
}

fn load_fitted() -> PolicyModel {
    let model = PolicyModel::fresh_on(10098998, PolicyDevice::Cpu).unwrap();
    TrainingArtifact::load_runtime_weights(&model, &output().join("weights")).unwrap();
    model
}

fn combat_outcome(model: &PolicyModel, physics: Physics) -> Metrics {
    let mut state = game(physics);
    for _ in 0..60 {
        if state.terminal {
            break;
        }
        let (frame, space) = state.prepare();
        state.action(model.choose(&frame, &space).unwrap().action);
    }
    state.total
}

fn trip_gates(parent: &PolicyModel, learned: &PolicyModel, report: &mut String) -> bool {
    let mut viable = [0, 0];
    let mut passed = [0, 0];
    let mut finish_viable = [0, 0];
    let mut finish_passed = [0, 0];
    for setup in trip::setups(false, &[1, 3]) {
        let (mut reference, reference_metrics) = trip::run(setup, trip::Controller::Reference);
        let valid = trip::valid_demonstration(setup, &reference);
        reference.rows.clear();
        writeln!(report, "held_reference setup={setup:?} viable={valid} trace={reference:?} metrics={reference_metrics:?}").unwrap();
        for (name, model) in [("parent", parent), ("learned", learned)] {
            let (mut trace, metrics) = trip::run(setup, trip::Controller::Neural(model));
            let pass = trip::valid_demonstration(setup, &trace);
            let rows = std::mem::take(&mut trace.rows);
            writeln!(report, "held_trip controller={name} setup={setup:?} viable={valid} pass={pass} trace={trace:?} metrics={metrics:?}").unwrap();
            for row in rows.into_iter().take(24) {
                writeln!(report, "{name}_trace {row}").unwrap();
            }
            if name == "learned" && valid {
                if setup.threat == trip::Threat::Finish {
                    finish_viable[setup.side] += 1;
                    finish_passed[setup.side] += u32::from(pass);
                } else {
                    viable[setup.side] += 1;
                    passed[setup.side] += u32::from(pass);
                }
            }
        }
    }
    writeln!(report, "trip_viable={viable:?} trip_passed={passed:?} finish_viable={finish_viable:?} finish_passed={finish_passed:?}").unwrap();
    (0..2).all(|side| {
        viable[side] > 0
            && passed[side] * 4 >= viable[side] * 3
            && finish_viable[side] > 0
            && finish_passed[side] == finish_viable[side]
    })
}

fn combat_gates(parent: &PolicyModel, learned: &PolicyModel, report: &mut String) -> bool {
    let mut count = [0, 0];
    let mut non_worse = [0, 0];
    let mut mango = [0, 0];
    let mut empty = true;
    for (coverage, physics) in combat_setups(false) {
        let before = combat_outcome(parent, physics);
        let after = combat_outcome(learned, physics);
        let gain = after.score - before.score;
        count[physics.side] += 1;
        non_worse[physics.side] += u32::from(gain >= -1e-4);
        if coverage == "mango_combat" {
            mango[physics.side] += u32::from(
                after.mango > 0 && after.damage > 0 && after.score >= before.score - 1e-4,
            );
        }
        if coverage == "empty" {
            empty &= after.casts == 0 && after.mana == 0;
        }
        writeln!(report, "held_combat coverage={coverage} physics={physics:?} parent={before:?} learned={after:?} gain={gain}").unwrap();
    }
    writeln!(report, "combat_count={count:?} non_worse={non_worse:?} mango_conversions={mango:?} empty_no_waste={empty}").unwrap();
    empty && (0..2).all(|side| non_worse[side] * 4 >= count[side] * 3 && mango[side] == 2)
}

fn ordinary_outcome(model: &PolicyModel, seed: u64, side: usize) -> (f64, f64, Metrics, u32, bool) {
    let mut state = ordinary(seed, side);
    let origin = state.seats[side].tracker.own_hero().unwrap().pos;
    let mid = Vec2::from_ints(9216, 9216);
    let mut early = 0.0;
    let mut lane = 0.0;
    for _ in 0..1500 {
        if state.terminal {
            break;
        }
        let (frame, space) = state.prepare();
        let action = model.choose(&frame, &space).unwrap().action;
        trip::act(&mut state, action, true, |state| {
            if let Some(hero) = state.seats[side].tracker.own_hero() {
                let progress = trip::distance(origin, mid) - trip::distance(hero.pos, mid);
                if state.arena.tick() == 901 {
                    early = progress;
                }
                if state.arena.tick() == 1801 {
                    lane = progress;
                }
            }
        });
    }
    (early, lane, state.total, state.arena.tick(), state.terminal)
}

#[test]
#[ignore = "Guarded matched parent-versus-learned M17 outcomes, no DEV/cohort games."]
fn matched_m17_outcomes() {
    let started = Instant::now();
    let parent = parent();
    let learned = load_fitted();
    let mut report = String::new();
    let trips = trip_gates(&parent, &learned, &mut report);
    let combat = combat_gates(&parent, &learned, &mut report);
    for side in 0..2 {
        let physics = Physics {
            side,
            own_hp: 100,
            enemy_hp: 680,
            mana: 353,
            mango: false,
            facing: if side == 0 { 0 } else { 32768 },
            ..mango_physics()
        };
        writeln!(
            report,
            "negative_mid_diagnostic_not_held side={side} parent={:?} learned={:?}",
            combat_outcome(&parent, physics),
            combat_outcome(&learned, physics)
        )
        .unwrap();
    }
    let mut opening = true;
    let mut activity = true;
    for seed in [10099980, 10099981] {
        for side in 0..2 {
            for (name, model) in [("parent", &parent), ("learned", &learned)] {
                let (early, lane, metrics, tick, terminal) = ordinary_outcome(model, seed, side);
                let active = metrics.damage > 0 || metrics.last_hits > 0;
                if name == "learned" {
                    opening &= early >= 1000.0 && lane >= 5000.0;
                    activity &= active;
                }
                writeln!(report, "ordinary controller={name} seed={seed} side={side} progress901={early} progress1801={lane} mid_activity={active} tick={tick} terminal={terminal} metrics={metrics:?}").unwrap();
            }
        }
    }
    writeln!(report, "trip_gate={trips}\ncombat_gate={combat}\nopening_gate={opening}\nmid_activity_gate={activity}\nreadiness={}\nqualified=false\ndev_games_run=0\nwall_seconds={:.3}", trips && combat && opening && activity, started.elapsed().as_secs_f64()).unwrap();
    write_new(&output().join("OUTCOMES.txt"), &report);
    eprintln!("{report}");
}

#[test]
#[ignore = "Read-only branch witness on fixed M17 models; no fitting or additional cohort."]
fn remaining_conversion_branch_witness() {
    let frozen = parent();
    let learned = load_fitted();
    let mut report = String::new();
    for (_, physics) in combat_setups(false)
        .into_iter()
        .filter(|(coverage, _)| *coverage == "mango_combat")
    {
        let mut state = game(physics);
        let (frame, space) = state.prepare();
        let parent_action = frozen.choose(&frame, &space).unwrap().action;
        let learned_action = learned.choose(&frame, &space).unwrap().action;
        let mut prefix = Prefix {
            physics,
            actions: vec![],
        };
        let ranked = neural_rank(&prefix, &frozen);
        let parent_suffix = neural_estimate(&prefix, learned_action, &frozen);
        let learned_suffix = neural_estimate(&prefix, learned_action, &learned);
        prefix.actions.push(learned_action);
        let mut next = rank::replay(&prefix);
        let (next_frame, next_space) = next.prepare();
        writeln!(report, "physics={physics:?} root_parent={parent_action:?} root_learned={learned_action:?} reference_best={:?} same_root_action_parent_suffix={} same_root_action_learned_suffix={} next_parent={:?} next_learned={:?} next_hero={:?}", ranked.best, parent_suffix.score, learned_suffix.score, frozen.choose(&next_frame, &next_space).unwrap().action, learned.choose(&next_frame, &next_space).unwrap().action, next.seats[physics.side].tracker.own_hero().map(|hero| (hero.hp, hero.mana, hero.pos))).unwrap();
    }
    write_new(&output().join("REMAINING.txt"), &report);
    eprintln!("{report}");
}

pub(super) fn file_hash(path: &Path) -> String {
    assert!(std::fs::metadata(path).unwrap().len() < 16 * 1024 * 1024);
    Sha256::digest(std::fs::read(path).unwrap())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[test]
#[ignore = "Read-only replay audit of frozen rehearsal cast labels, no training or new cohort."]
fn audit_rehearsal_cast_evidence() {
    let mut report = String::new();
    let mut total = 0;
    let mut unsupported = 0;
    for seed in [10098980, 10098981] {
        for side in 0..2 {
            let mut state = ordinary(seed, side);
            for _ in 0..1500 {
                if state.terminal {
                    break;
                }
                let before = state.total.clone();
                let hero = state.seats[side].tracker.own_hero().map(|unit| unit.id);
                let enemy = state.seats[side].tracker.current().unwrap().players[1 - side].unit;
                let (_, space, action, request) = teacher_sample(&mut state);
                let other = teacher_request(&mut state.seats[1 - side]).unwrap();
                let mut magical_hero = false;
                let mut magical_targets = Vec::new();
                let mut paid = Vec::new();
                for tick in 0..3 {
                    if state.terminal {
                        break;
                    }
                    state.tick(
                        if tick == 0 { request } else { None },
                        if tick == 0 { other } else { None },
                    );
                    magical_hero |=
                        magic_evidence(&state, hero, enemy, &mut magical_targets, &mut paid);
                }
                if matches!(
                    action,
                    StructuredAction::Cast {
                        unit: ControlledUnit::Hero,
                        ..
                    }
                ) && state.total.casts > before.casts
                    && (state.total.damage > before.damage || state.total.gold > before.gold)
                {
                    let supported =
                        magical_hero || paid.iter().any(|unit| magical_targets.contains(unit));
                    total += 1;
                    unsupported += u32::from(!supported);
                    writeln!(report, "seed={seed} side={side} tick={} supported_magic_hit_or_paid_kill={supported}", space.tick()).unwrap();
                }
            }
        }
    }
    writeln!(
        report,
        "cast_labels={total} unsupported={unsupported} model_updates=0"
    )
    .unwrap();
    write_new(&output().join("REHEARSAL_AUDIT.txt"), &report);
    eprintln!("{report}");
    assert_eq!(
        unsupported, 0,
        "rehearsal evidence audit found unsupported cast anchors"
    );
}

fn magic_evidence(
    state: &Game,
    hero: Option<bota_proto::EntityId>,
    enemy: Option<bota_proto::EntityId>,
    targets: &mut Vec<bota_proto::EntityId>,
    paid: &mut Vec<bota_proto::EntityId>,
) -> bool {
    assert!(targets.len() <= 192);
    assert!(paid.len() <= 192);
    let mut hit = false;
    for event in state.seats[state.side]
        .tracker
        .recent_events()
        .iter()
        .filter(|event| event.tick == state.arena.tick())
    {
        match event.kind {
            EventKind::Damaged {
                source,
                target,
                kind: DamageKind::Magical,
                amount,
                ..
            } if source == hero && amount > 0 => {
                hit |= Some(target) == enemy;
                targets.push(target);
            }
            EventKind::Died {
                unit, killer, gold, ..
            } if killer == hero && gold > 0 => paid.push(unit),
            _ => {}
        }
    }
    hit
}

pub(super) fn write_new(path: &Path, text: &str) {
    use std::io::Write;
    assert!(text.len() < 16 * 1024 * 1024);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    file.write_all(text.as_bytes()).unwrap();
    file.sync_all().unwrap();
}
