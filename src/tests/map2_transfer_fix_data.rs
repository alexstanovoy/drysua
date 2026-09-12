use super::*;
use crate::model::CheckedBehavioralSample;
use crate::ppo_arena::map2_skill_bootstrap as skills;
use crate::{ControlledUnit, EntityIndex, PointIndex};
use bota_proto::UnitKind;
use bota_server::game::{Level, rules};

pub(crate) struct Row {
    pub sample: CheckedBehavioralSample,
    pub space: ActionSpace,
    pub action: StructuredAction,
    pub seed: u64,
    pub side: usize,
    pub origin: &'static str,
}

impl Row {
    fn new(
        frame: FeatureFrame,
        space: ActionSpace,
        action: StructuredAction,
        seed: u64,
        side: usize,
        origin: &'static str,
    ) -> Self {
        assert!(side < 2);
        let sample = CheckedBehavioralSample::new(frame, &space, action)
            .expect("checked pre-action frame/label/legality");
        Self {
            sample,
            space,
            action,
            seed,
            side,
            origin,
        }
    }
}

pub(super) fn new_specs(training: bool) -> Vec<skills::Spec> {
    skills::specs(!training, if training { &[0, 2] } else { &[1, 3] })
        .into_iter()
        .map(|mut spec| {
            spec.seed += 200;
            spec
        })
        .collect()
}

pub(super) fn skill_environment(spec: skills::Spec) -> skills::Environment {
    let (mut arena, mut start) = Arena::new(ArenaConfig {
        seats: 2,
        map: MapId(2),
        seed: spec.seed,
    })
    .unwrap();
    let mut goal = Vec2::from_ints(0, 0);
    let configured = arena.configure_for_test(|world| {
        skills::configure(world, spec);
        world.tick = [901, 1051, 1501, 1651][spec.variant as usize];
        if spec.variant >= 2 {
            for side in 0..2 {
                let hero = world.seats[side].unit.unwrap();
                world.seats[side].level = 4;
                world.seats[side].xp = rules::XP_THRESHOLDS[3];
                world.level.insert(hero, Level(4));
                world.abilities.get_mut(hero).unwrap().slots[3].level = 2;
            }
        }
        if spec.kind != skills::Kind::Supply {
            world.seats[spec.side].gold = [0, 5, 20, 30][spec.variant as usize];
        }
        world.settle();
        if spec.kind == skills::Kind::MangoFull {
            world.fill_pools(world.seats[spec.side].unit.unwrap());
        }
        goal = world.map.fountains[spec.side];
    });
    for (messages, fresh) in start.messages.iter_mut().zip(configured.messages) {
        messages.truncate(1);
        messages.extend(fresh);
    }
    let seats = setup_seats(start).unwrap();
    let start_distance = skills::distance(seats[spec.side].tracker.own_hero().unwrap().pos, goal);
    skills::Environment {
        arena,
        seats,
        spec,
        goal,
        start_distance,
        result: skills::ResultCounts::default(),
    }
}

fn first_order(
    environment: &skills::Environment,
    space: &ActionSpace,
    decision: u32,
) -> Option<StructuredAction> {
    use skills::Kind;
    if !matches!(
        environment.spec.kind,
        Kind::Recovery | Kind::Finish | Kind::CreepHealthy | Kind::NoMana
    ) {
        return None;
    }
    if decision != 0 {
        return Some(StructuredAction::Continue);
    }
    if environment.spec.kind == Kind::Recovery {
        let point = space
            .point_candidates()
            .iter()
            .enumerate()
            .filter(|(index, _)| space.move_point_mask(ControlledUnit::Hero)[*index])
            .min_by_key(|(_, point)| point.position.distance_squared(environment.goal))
            .unwrap()
            .0;
        return Some(StructuredAction::MovePoint {
            unit: ControlledUnit::Hero,
            point: PointIndex(point),
        });
    }
    let target = space
        .entity_candidates()
        .iter()
        .position(|unit| {
            unit.relation == crate::EntityRelation::Enemy
                && unit.kind
                    == if environment.spec.kind == Kind::CreepHealthy {
                        UnitKind::CreepMelee
                    } else {
                        UnitKind::Hero
                    }
        })
        .unwrap();
    Some(StructuredAction::AttackUnit {
        unit: ControlledUnit::Hero,
        target: EntityIndex(target),
    })
}

pub(super) fn collect_skills(training: bool) -> Vec<Row> {
    let mut rows = Vec::new();
    for spec in new_specs(training) {
        let mut environment = skill_environment(spec);
        let mut trajectory = Vec::new();
        for decision in 0..60 {
            let (frame, space) =
                prepare_neural_seat_policy_sample(&mut environment.seats[spec.side]).unwrap();
            let action = first_order(&environment, &space, decision)
                .unwrap_or_else(|| skills::script(&environment, &space));
            let request =
                neural_policy_request_in_space(&mut environment.seats[spec.side], action, &space)
                    .unwrap()
                    .1;
            if action != StructuredAction::Continue || [0, 1, 2, 5, 15, 30, 50].contains(&decision)
            {
                trajectory.push(Row::new(
                    frame,
                    space,
                    action,
                    spec.seed,
                    spec.side,
                    "validated_skill_trajectory",
                ));
            }
            for tick in 0..3 {
                environment.advance(if tick == 0 { request } else { None });
            }
        }
        assert!(
            skills::completed(spec.kind, &environment.result),
            "unvalidated {spec:?}: {:?}",
            environment.result
        );
        assert_eq!(environment.seats[spec.side].rejections, 0);
        assert!(trajectory.len() <= 20);
        rows.extend(trajectory);
    }
    assert!(rows.len() <= 1024);
    rows
}

pub(super) fn old_regression_rows() -> Vec<Row> {
    skills::specs(true, &[1])
        .into_iter()
        .flat_map(skills::collect)
        .map(|row| {
            Row::new(
                row.frame,
                row.space,
                row.action,
                row.spec.seed,
                row.spec.side,
                "old_validation_regression_not_new_holdout",
            )
        })
        .collect()
}

fn teacher_decision(
    seat: &mut ArenaSeatPolicy,
) -> (FeatureFrame, ActionSpace, StructuredAction, Option<Request>) {
    let (frame, _) = prepare_neural_observer_sample(seat).unwrap();
    let (action, space) = seat
        .teacher
        .decide(&seat.tracker, &seat.persistence, &seat.readiness)
        .unwrap();
    assert!(frame.matches_action_space(&space));
    seat.local
        .note_decision(space.tick(), action.kind())
        .unwrap();
    let decoded = space.decode(action).unwrap();
    let request = issue_request(seat, decoded, &space, action.kind(), true).unwrap();
    (frame, space, action, request)
}

pub(crate) fn collect_anchor(seed: u64, side: usize) -> Vec<Row> {
    let mut environment = opening(seed, side);
    let mut rows = Vec::new();
    for decision in 0..OPENING_TICKS / 3 {
        let before = environment.seats[side].tracker.own_hero().unwrap().clone();
        let (frame, space, action, request) = teacher_decision(&mut environment.seats[side]);
        let active = environment.seats[side].local.active_order();
        let opponent = teacher_request(&mut environment.seats[1 - side]).unwrap();
        for tick in 0..3 {
            environment.advance(
                if tick == 0 { request } else { None },
                if tick == 0 { opponent } else { None },
            );
        }
        let Some(after) = environment.seats[side].tracker.own_hero() else {
            continue;
        };
        let useful_progress = distance(before.pos, Vec2::from_ints(9216, 9216))
            - distance(after.pos, Vec2::from_ints(9216, 9216))
            > 0.5;
        let verified = match action {
            StructuredAction::Learn { slot } => after.abilities[slot.0 as usize].level > before.abilities[slot.0 as usize].level,
            StructuredAction::Buy { .. } => request.is_some() && environment.seats[side].tracker.recent_events().iter().any(|event| event.tick > space.tick() && matches!(event.kind, EventKind::ItemBought { slot, .. } if slot == SlotId(side as u8))),
            StructuredAction::MovePoint { unit: ControlledUnit::Hero, .. } | StructuredAction::AttackMovePoint { unit: ControlledUnit::Hero, .. } => useful_progress && request.is_some(),
            StructuredAction::Continue => useful_progress && decision % 20 == 0 && active.is_some_and(|order| matches!(order.kind, ActionKind::MovePoint | ActionKind::AttackMovePoint)),
            _ => false,
        };
        if verified {
            rows.push(Row::new(
                frame,
                space,
                action,
                seed,
                side,
                "unchanged_teacher_native_opening",
            ));
        }
    }
    assert_eq!(environment.seats[side].rejections, 0);
    assert!(rows.len() <= 512);
    assert!(!rows.is_empty());
    rows
}

pub(super) fn opening_specs(training: bool) -> Vec<(u64, usize)> {
    let base = if training { 10092080 } else { 10093080 };
    [base, base + 1]
        .into_iter()
        .flat_map(|seed| (0..2).map(move |side| (seed, side)))
        .collect()
}

pub(super) fn collect_anchors(training: bool) -> Vec<Row> {
    opening_specs(training)
        .into_iter()
        .flat_map(|(seed, side)| collect_anchor(seed, side))
        .collect()
}

pub(super) fn skill_outcomes(model: &PolicyModel, report: &mut String, label: &str) -> [u32; 2] {
    skill_outcomes_for(model, &new_specs(false), true, report, label)
}

pub(super) fn old_skill_outcomes(
    model: &PolicyModel,
    report: &mut String,
    label: &str,
) -> [u32; 2] {
    skill_outcomes_for(model, &skills::specs(true, &[1]), false, report, label)
}

fn skill_outcomes_for(
    model: &PolicyModel,
    specs: &[skills::Spec],
    varied: bool,
    report: &mut String,
    label: &str,
) -> [u32; 2] {
    let mut passes = [0, 0];
    for &spec in specs {
        let mut environment = if varied {
            skill_environment(spec)
        } else {
            skills::environment(spec)
        };
        for _ in 0..60 {
            let (frame, space) =
                prepare_neural_seat_policy_sample(&mut environment.seats[spec.side]).unwrap();
            let action = model.choose(&frame, &space).unwrap().action;
            let request =
                neural_policy_request_in_space(&mut environment.seats[spec.side], action, &space)
                    .unwrap()
                    .1;
            for tick in 0..3 {
                environment.advance(if tick == 0 { request } else { None });
            }
        }
        let pass = skills::completed(spec.kind, &environment.result)
            && environment.seats[spec.side].rejections == 0;
        passes[spec.side] += u32::from(pass);
        writeln!(
            report,
            "{label} skill={spec:?} pass={pass} rejections={} outcome={:?}",
            environment.seats[spec.side].rejections, environment.result
        )
        .unwrap();
    }
    passes
}
