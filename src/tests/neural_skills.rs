use super::*;
use crate::{ControlledUnit, EntityIndex, ImitationSample, StructuredAction};
use bota_proto::{Fixed, UnitKind, Vec2};
use bota_server::game::UnitOrder;

#[derive(Clone, Copy)]
enum SkillController<'model> {
    Attack,
    Idle,
    Neural(&'model PolicyModel),
    Teacher,
}

fn finish_fixture(side: usize, distance: i32) -> (Arena, Vec<ArenaSeatPolicy>) {
    finish_fixture_seeded(side, distance, 10084000)
}

fn finish_fixture_seeded(side: usize, distance: i32, seed: u64) -> (Arena, Vec<ArenaSeatPolicy>) {
    assert!(side < 2);
    assert!((120..=520).contains(&distance));
    let (mut arena, mut start) = Arena::new(ArenaConfig {
        seats: 2,
        map: MapId(0),
        seed,
    })
    .expect("skill arena");
    let configured = arena.configure_for_test(|world| {
        let heroes = [
            world.seats[0].unit.expect("Radiant"),
            world.seats[1].unit.expect("Dire"),
        ];
        let removed: Vec<_> = world
            .entities
            .iter()
            .filter(|entity| {
                !heroes.contains(entity) && world.kind.get(*entity) != Some(&UnitKind::Ancient)
            })
            .collect();
        assert!(removed.len() < 4096);
        for entity in removed {
            assert!(world.despawn(entity));
        }
        for (index, hero) in heroes.into_iter().enumerate() {
            world.seats[index].gold = 0;
            world.statuses.remove(hero);
            world.set_order(hero, UnitOrder::Stand);
            let position = world.transform.get_mut(hero).expect("hero transform");
            position.pos = Vec2::from_ints(8600 + index as i32 * distance, 8900);
            position.facing.brads = if index == 0 { 0 } else { 32768 };
            world.health.get_mut(hero).expect("health").hp =
                Fixed::from_int(if index == side { 80 } else { 20 });
            world.mana.get_mut(hero).expect("mana").mana = Fixed::ZERO;
            world.abilities.get_mut(hero).expect("abilities").slots[3].level = 1;
        }
    });
    for (messages, fresh) in start.messages.iter_mut().zip(configured.messages) {
        messages.truncate(1);
        messages.extend(fresh);
    }
    (arena, setup_seats(start).expect("projected seats"))
}

fn run_finish_skill(
    controller: SkillController<'_>,
    side: usize,
    distance: i32,
) -> (u64, [u32; ActionKind::COUNT]) {
    let (mut arena, mut seats) = finish_fixture(side, distance);
    let mut kinds = [0; ActionKind::COUNT];
    for decision in 0..50 {
        let (frame, space) = prepare_skill_controller_sample(&mut seats[side], controller);
        let enemy = space.entity_candidates().iter().position(|entity| {
            entity.kind == UnitKind::Hero && entity.relation == crate::EntityRelation::Enemy
        });
        let action = match controller {
            SkillController::Idle => StructuredAction::Continue,
            SkillController::Attack if decision == 0 => StructuredAction::AttackUnit {
                unit: ControlledUnit::Hero,
                target: EntityIndex(enemy.expect("visible enemy")),
            },
            SkillController::Attack => StructuredAction::Continue,
            SkillController::Neural(model) => {
                model.choose(&frame, &space).expect("pure neural").action
            }
            SkillController::Teacher => {
                let seat = &mut seats[side];
                seat.teacher
                    .decide(&seat.tracker, &seat.persistence, &seat.readiness)
                    .expect("frozen expert")
                    .0
            }
        };
        assert!(space.allows(action));
        kinds[action.kind().index()] += 1;
        let (_, request) =
            neural_policy_request_in_space(&mut seats[side], action, &space).expect("request");
        if matches!(controller, SkillController::Teacher) {
            note_skill_teacher_request(&mut seats[side], request, space.tick());
        }
        advance_skill(&mut arena, &mut seats, side, request);
    }
    let kills = seats[side]
        .tracker
        .latest_summary()
        .expect("summary")
        .allied
        .kills;
    eprintln!("finish_skill side={side} distance={distance} kills={kills} kinds={kinds:?}");
    (kills, kinds)
}

fn prepare_skill_controller_sample(
    seat: &mut ArenaSeatPolicy,
    controller: SkillController<'_>,
) -> (FeatureFrame, ActionSpace) {
    let sample = if matches!(controller, SkillController::Neural(_)) {
        prepare_neural_seat_policy_sample(seat).expect("Candidate skill frame")
    } else {
        prepare_seat_policy_sample(seat).expect("legacy skill frame")
    };
    assert_eq!(
        seat.order_bookkeeping.is_candidate(),
        matches!(controller, SkillController::Neural(_))
    );
    assert_eq!(
        seat.order_bookkeeping.is_neural(),
        matches!(controller, SkillController::Neural(_))
    );
    sample
}

fn prepare_skill_teacher_sample(seat: &mut ArenaSeatPolicy) -> (FeatureFrame, ActionSpace) {
    let sample = prepare_neural_observer_sample(seat).expect("Observer pre-action frame");
    assert!(seat.order_bookkeeping.is_neural());
    assert!(!seat.order_bookkeeping.is_candidate());
    sample
}

#[test]
fn matched_contract_skill_neural_is_candidate_and_controls_stay_legacy() {
    let model = PolicyModel::fresh(10084001).expect("model");
    for side in 0..2 {
        for controller in [
            SkillController::Neural(&model),
            SkillController::Teacher,
            SkillController::Attack,
            SkillController::Idle,
        ] {
            let (_, mut seats) = finish_fixture(side, 240);
            prepare_skill_controller_sample(&mut seats[side], controller);
            assert_eq!(
                seats[side].order_bookkeeping.is_candidate(),
                matches!(controller, SkillController::Neural(_))
            );
            assert_eq!(
                seats[side].order_bookkeeping.is_neural(),
                matches!(controller, SkillController::Neural(_))
            );
        }
    }
}

#[test]
fn matched_contract_skill_teacher_annotations_are_observers_not_candidates() {
    for side in 0..2 {
        let (_, mut seats) = finish_fixture(side, 240);
        prepare_skill_teacher_sample(&mut seats[side]);
        assert!(seats[side].order_bookkeeping.is_neural());
        assert!(!seats[side].order_bookkeeping.is_candidate());
    }
}

#[test]
fn matched_contract_skill_observer_annotations_preserve_legacy_teacher_transport() {
    for side in 0..2 {
        let (mut source_arena, mut source) = finish_fixture(side, 240);
        let (mut target_arena, mut target) = finish_fixture(side, 240);
        for _ in 0..32 {
            let (source_frame, _) =
                prepare_seat_policy_sample(&mut source[side]).expect("legacy frame");
            let (target_frame, _) = prepare_skill_teacher_sample(&mut target[side]);
            let mut requests = [None, None];
            let mut identities = Vec::with_capacity(2);
            let mut actions = Vec::with_capacity(2);
            for (index, (seat, frame)) in [
                (&mut source[side], source_frame),
                (&mut target[side], target_frame),
            ]
            .into_iter()
            .enumerate()
            {
                let (action, space) = seat
                    .teacher
                    .decide(&seat.tracker, &seat.persistence, &seat.readiness)
                    .expect("unchanged expert");
                identities.push(
                    SampleIdentity::from_frame(
                        SeedNamespace::Training,
                        10084000,
                        0,
                        space.tick(),
                        &frame,
                    )
                    .expect("identity"),
                );
                actions.push(action);
                requests[index] = neural_policy_request_in_space(seat, action, &space)
                    .expect("original transport")
                    .1;
                note_skill_teacher_request(seat, requests[index], space.tick());
            }
            assert_eq!(identities[0], identities[1]);
            assert_eq!(actions[0], actions[1]);
            assert_eq!(requests[0], requests[1]);
            assert_eq!(source[side].teacher, target[side].teacher);
            assert_eq!(source[side].persistence, target[side].persistence);
            assert_eq!(source[side].readiness, target[side].readiness);
            advance_skill(&mut source_arena, &mut source, side, requests[0]);
            advance_skill(&mut target_arena, &mut target, side, requests[1]);
            assert_eq!(
                source[side].tracker.latest_summary(),
                target[side].tracker.latest_summary()
            );
        }
    }
}

#[test]
fn frozen_teacher_solves_the_executable_finish_fixtures() {
    for side in 0..2 {
        for distance in [240, 480] {
            assert_eq!(
                run_finish_skill(SkillController::Teacher, side, distance).0,
                1
            );
        }
    }
}

fn collect_finish_skill_rows(side: usize, distance: i32, seed: u64) -> Vec<ImitationSample> {
    let (mut arena, mut seats) = finish_fixture_seeded(side, distance, seed);
    let mut samples = Vec::with_capacity(32);
    for _ in 0..32 {
        let seat = &mut seats[side];
        let (frame, _) = prepare_skill_teacher_sample(seat);
        let (action, space) = seat
            .teacher
            .decide(&seat.tracker, &seat.persistence, &seat.readiness)
            .expect("frozen expert");
        let identity =
            SampleIdentity::from_frame(SeedNamespace::Training, seed, 0, space.tick(), &frame)
                .expect("identity");
        samples
            .push(ImitationSample::teacher(frame, &space, action, identity).expect("exact label"));
        let (_, request) =
            neural_policy_request_in_space(seat, action, &space).expect("teacher command");
        note_skill_teacher_request(seat, request, space.tick());
        advance_skill(&mut arena, &mut seats, side, request);
    }
    assert_eq!(samples.len(), 32);
    assert_eq!(
        seats[side]
            .tracker
            .latest_summary()
            .expect("summary")
            .allied
            .kills,
        1,
        "demonstrator did not finish: side={side} distance={distance} seed={seed} actions={:?}",
        samples
            .iter()
            .map(ImitationSample::teacher_action)
            .collect::<Vec<_>>()
    );
    samples
}

#[test]
#[ignore = "Bounded synthetic-scene BC hypothesis probe; never a gameplay release"]
fn learn_executable_finish_skill_without_architecture_change() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let output = root.join("artifacts/temp/neural-reset-20260908/finish-skill-bc");
    assert!(!output.exists());
    let pool = finish_skill_pool();
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    let device = PolicyDevice::Cuda { ordinal: 0 };
    #[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
    let device = PolicyDevice::Cpu;
    let model = PolicyModel::fresh_on(10084110, device).expect("model");
    TrainingArtifact::load_runtime_weights(
        &model,
        &root.join("artifacts/temp/input-facts-m12-u10-init"),
    )
    .expect("initializer");
    train_finish_skill_model(&model, &pool);
    std::fs::create_dir(&output).expect("diagnostic directory");
    TrainingArtifact::save_runtime_weights(&model, &output).expect("new diagnostic weights");
    std::fs::write(output.join("PROVENANCE.txt"), "synthetic_test_scenes=true\nfull_match_qualification=false\nsource=adbecb8293548ad1602b24b046b9a6f032fc62798d46102029a1f3448d764bdd\noptimizer=fresh\ntrain_distances=120,200,360,520\nvalidation_distances=240,480\ntrain_seeds=10084100..10084103\nvalidation_seed=10084000\nepochs=128\nlearning_rate=3e-4\n").expect("provenance");
    let actor = PolicyModel::fresh(10084112).expect("CPU evaluation");
    actor
        .import_parameters(&model.export_parameters().expect("trained weights"))
        .expect("actor weights");
    assert_eq!(
        finish_skill_successes(&actor),
        4,
        "held-distance skill transfer"
    );
}

fn finish_skill_pool() -> ImitationPool {
    let seeds: Vec<_> = (10084100..10084104).collect();
    let namespaces = SeedNamespaces::new(seeds.clone(), vec![10084000], vec![10084200])
        .expect("disjoint sources");
    let mut pool = ImitationPool::new(
        256,
        10084110,
        namespaces,
        TrainingScope::new(MapId(0), IMITATION_RULES_AUDIT_VERSION).expect("scope"),
    )
    .expect("pool");
    for (seed, distance) in seeds.into_iter().zip([120, 200, 360, 520]) {
        for side in 0..2 {
            for sample in collect_finish_skill_rows(side, distance, seed) {
                assert!(pool.push(sample).expect("sample").is_none());
            }
        }
    }
    assert_eq!(pool.statistics().teacher, 256);
    assert_eq!(pool.statistics().dagger, 0);
    pool
}

fn train_finish_skill_model(model: &PolicyModel, pool: &ImitationPool) {
    assert_eq!(pool.statistics().teacher, 256);
    assert_eq!(pool.statistics().dagger, 0);
    let mut trainer = BehavioralTrainer::new(
        64,
        10084111,
        AdamConfig {
            learning_rate: 3e-4,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            gradient_clip: 0.5,
        },
        model,
        pool,
    )
    .expect("fresh optimizer");
    let started = Instant::now();
    for epoch in 1..=128 {
        let report = trainer.train_epoch(model, pool).expect("BC epoch");
        if epoch % 32 == 0 {
            eprintln!(
                "finish_bc epoch={epoch} loss={} seconds={:.3}",
                report.average_loss,
                started.elapsed().as_secs_f64()
            );
        }
        assert!(started.elapsed() < Duration::from_secs(600));
    }
}

fn finish_skill_successes(model: &PolicyModel) -> u64 {
    assert_eq!(model.parameter_count(), crate::MODEL_PARAMETER_COUNT);
    let mut successes = 0;
    for side in 0..2 {
        for distance in [240, 480] {
            successes += run_finish_skill(SkillController::Neural(model), side, distance).0;
        }
    }
    assert!(successes <= 4);
    successes
}

fn note_skill_teacher_request(seat: &mut ArenaSeatPolicy, request: Option<Request>, tick: u32) {
    assert!(tick > 0);
    assert!(tick <= 151);
    if let Some(request) = request {
        seat.teacher.note_sent(
            request.seq,
            crate::IssuedOrder {
                unit: request.unit,
                order: request.order,
            },
            tick,
        );
    }
}

fn advance_skill(
    arena: &mut Arena,
    seats: &mut [ArenaSeatPolicy],
    side: usize,
    request: Option<Request>,
) {
    assert!(side < seats.len());
    assert_eq!(arena.seat_count(), seats.len());
    for tick in 0..3 {
        let mut requests = [None, None];
        if tick == 0 {
            requests[side] = request;
        }
        let step = arena.step(&requests).expect("skill tick");
        for (seat, messages) in seats.iter_mut().zip(step.messages) {
            assert!(
                observe_messages(seat, &messages)
                    .expect("seat messages")
                    .is_none()
            );
            assert_eq!(seat.rejections, 0);
        }
    }
}

#[test]
fn attack_then_continue_finishes_a_visible_enemy_but_idle_does_not() {
    for side in 0..2 {
        for distance in [240, 480] {
            assert_eq!(
                run_finish_skill(SkillController::Attack, side, distance).0,
                1
            );
            assert_eq!(run_finish_skill(SkillController::Idle, side, distance).0, 0);
        }
    }
}

#[test]
#[ignore = "Retained-weight control-skill reproduction; not full-match qualification"]
fn retained_neural_policy_finishes_all_executable_low_health_duels() {
    let weights = std::env::var("DRYSUA_SKILL_WEIGHTS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/temp/input-facts-m12-u10-init")
        });
    let model = PolicyModel::fresh(10084001).expect("CPU model");
    TrainingArtifact::load_runtime_weights(&model, &weights).expect("retained weights");
    let mut successes = 0;
    for side in 0..2 {
        for distance in [240, 480] {
            successes += run_finish_skill(SkillController::Neural(&model), side, distance).0;
        }
    }
    assert_eq!(
        successes, 4,
        "every case is executable without new movement primitives"
    );
}

#[test]
#[ignore = "Bounded interpolation of an audited skill correction; not a release gate"]
fn probe_skill_correction_interpolation() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let model = PolicyModel::fresh(10084120).expect("model");
    TrainingArtifact::load_runtime_weights(
        &model,
        &root.join("artifacts/temp/input-facts-m12-u10-init"),
    )
    .expect("original");
    let original = model.export_parameters().expect("original parameters");
    TrainingArtifact::load_runtime_weights(
        &model,
        &root.join("artifacts/temp/neural-reset-20260908/finish-skill-bc"),
    )
    .expect("corrected");
    let corrected = model.export_parameters().expect("corrected parameters");
    assert_eq!(original.len(), corrected.len());
    for (label, alpha) in [("025", 0.25f32), ("050", 0.5), ("075", 0.75)] {
        #[allow(
            clippy::float_arithmetic,
            reason = "Explicit F32 interpolation of two named artifacts"
        )]
        let values: Vec<_> = original
            .iter()
            .zip(&corrected)
            .map(|(source, target)| source + alpha * (target - source))
            .collect();
        model
            .import_parameters(&values)
            .expect("finite interpolation");
        let mut successes = 0;
        for side in 0..2 {
            for distance in [240, 480] {
                successes += run_finish_skill(SkillController::Neural(&model), side, distance).0;
            }
        }
        let output = root.join(format!(
            "artifacts/temp/neural-reset-20260908/finish-skill-alpha-{label}"
        ));
        std::fs::create_dir(&output).expect("new artifact directory");
        TrainingArtifact::save_runtime_weights(&model, &output).expect("new initialization");
        std::fs::write(output.join("PROVENANCE.txt"), format!("method=parameter_interpolation\nsource=input-facts-m12-u10-init\ntarget=finish-skill-bc\nalpha={alpha}\nvalidation_skill_success={successes}/4\nqualified=false\n")).expect("provenance");
        eprintln!("skill_interpolation alpha={alpha} successes={successes}/4");
    }
}
