use super::*;

#[path = "map2_recovery_004_audit.rs"]
mod audit;
#[path = "map2_recovery_004_tests.rs"]
mod tests;
#[path = "map2_recovery_004_data.rs"]
mod data;
#[path = "map2_recovery_004_fit.rs"]
mod fit;
#[path = "map2_recovery_004_eval.rs"]
mod evaluation;
#[path = "map2_recovery_004_witness.rs"]
mod witness4;

#[derive(Debug)]
struct Support {
    value: f64,
    survived: bool,
    productive: bool,
}

fn retain_reference(support: &Support) -> bool {
    assert!(support.value.is_finite());
    support.survived && support.productive && support.value >= -TIE
}

fn fountain_center(state: &Game) -> Vec2 {
    let tracker = &state.seats[state.side].tracker;
    tracker
        .current()
        .unwrap()
        .units
        .iter()
        .find(|unit| unit.kind == UnitKind::Fountain && unit.team == tracker.team())
        .expect("native own fountain in allowed seat view")
        .pos
}

fn full_inside(state: &Game) -> bool {
    state.seats[state.side]
        .tracker
        .own_hero()
        .is_some_and(|hero| {
            hero.hp >= hero.max_hp
                && hero.mana >= hero.max_mana
                && hero.pos.within(
                    fountain_center(state),
                    Fixed::from_int(rules::FOUNTAIN_HEAL_RADIUS),
                )
        })
}

fn data_reference(state: &Game, space: &ActionSpace) -> StructuredAction {
    if !full_inside(state) {
        return trip::reference(state, space);
    }
    let hero = state.seats[state.side].tracker.own_hero().unwrap();
    let (_, landing) = trip::fountain(space);
    let active = state.seats[state.side].local.active_order();
    if active.is_some_and(|order| {
        matches!(order.target, crate::ActivePolicyTarget::Point(point)
        if trip::distance(point, landing) >= 1.0 && trip::distance(hero.pos, point) > 1.0)
    }) {
        return StructuredAction::Continue;
    }
    let point = space
        .point_candidates()
        .iter()
        .enumerate()
        .filter(|(index, _)| space.move_point_mask(ControlledUnit::Hero)[*index])
        .min_by_key(|(_, point)| point.position.distance_squared(Vec2::from_ints(9216, 9216)))
        .expect("legal public midward departure point")
        .0;
    let action = StructuredAction::MovePoint {
        unit: ControlledUnit::Hero,
        point: PointIndex(point),
    };
    assert!(space.allows(action));
    assert!(full_inside(state));
    action
}

fn support_after(
    before: &Game,
    after: &Game,
    action: StructuredAction,
    space: &ActionSpace,
) -> Support {
    let tracker = &before.seats[before.side].tracker;
    let end = &after.seats[after.side].tracker;
    let survived = end.own_player().unwrap().deaths == tracker.own_player().unwrap().deaths;
    let productive = after.total.damage > before.total.damage
        || after.total.gold > before.total.gold
        || after.total.last_hits > before.total.last_hits;
    let progress = tracker
        .own_hero()
        .zip(end.own_hero())
        .is_some_and(|(source, target)| {
            if source.id != target.id {
                return false;
            }
            let active = before.seats[before.side].local.active_order();
            let toward = match action {
                StructuredAction::MovePoint { point, .. }
                | StructuredAction::AttackMovePoint { point, .. } => {
                    Some(space.point_candidates()[point.0].position)
                }
                StructuredAction::Continue => active.and_then(|order| match order.target {
                    crate::ActivePolicyTarget::Point(point) => Some(point),
                    _ => None,
                }),
                _ => None,
            };
            let walking = toward.is_some_and(|point| {
                trip::distance(source.pos, point) - trip::distance(target.pos, point) >= 30.0
            });
            let healing = source.pos.within(
                fountain_center(before),
                Fixed::from_int(rules::FOUNTAIN_HEAL_RADIUS),
            ) && target.pos.within(
                fountain_center(after),
                Fixed::from_int(rules::FOUNTAIN_HEAL_RADIUS),
            ) && (target.hp - source.hp >= source.max_hp / 8
                || target.mana - source.mana >= source.max_mana / 8);
            walking || healing
        });
    Support {
        value: after.total.score - before.total.score,
        survived,
        productive: productive || progress,
    }
}

fn reference_support(prefix: &Prefix, reference: &PolicyModel) -> Support {
    let before = rank::replay(prefix);
    let mut branch = rank::replay(prefix);
    let (frame, space) = branch.prepare();
    let action = reference.choose(&frame, &space).unwrap().action;
    branch.action(action);
    for _ in 1..HORIZON / 3 {
        if branch.terminal {
            break;
        }
        let (frame, space) = branch.prepare();
        branch.action(reference.choose(&frame, &space).unwrap().action);
    }
    support_after(&before, &branch, action, &space)
}

fn supported_target(
    prefix: &Prefix,
    reference: &PolicyModel,
) -> (Option<Row>, StructuredAction, bool, Support) {
    let (mut row, action, improved) = target_row(prefix, reference);
    let support = reference_support(prefix, reference);
    let retain = improved || retain_reference(&support);
    if !improved {
        row.identity = row
            .identity
            .replace("verified_reference_tie", "supported_reference_tie");
    }
    row.identity.push_str(&format!(":support={support:?}"));
    (retain.then_some(row), action, improved, support)
}

fn trip_setups4(training: bool) -> Vec<trip::Setup> {
    trip::setups(training, if training { &[0, 2] } else { &[1, 3] })
        .into_iter()
        .map(|mut setup| {
            setup.seed += 4000;
            setup
        })
        .collect()
}

fn combat_cases4(training: bool) -> Vec<(&'static str, Physics)> {
    combat_setups(training)
        .into_iter()
        .map(|(coverage, mut physics)| {
            physics.seed += 4000;
            (coverage, physics)
        })
        .collect()
}

const ROOT4: &str = "artifacts/temp/map2-gameplay-fix-20260912/recovery-004";
const INITIAL_SHA: &str = "ce20adc53cb9087f01a4acd7ea5447ce6b45873a2ed32f7b5518614b9382fddf";

fn root4() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(ROOT4)
}

fn initial() -> PolicyModel {
    assert_eq!(
        file_hash(&root3().join("weights/drysua.weights.safetensors")),
        INITIAL_SHA
    );
    let model = PolicyModel::fresh_on(10102999, PolicyDevice::Cpu).unwrap();
    TrainingArtifact::load_runtime_weights(&model, &root3().join("weights")).unwrap();
    assert_eq!(crate::MODEL_SCHEMA_VERSION, 17);
    model
}
