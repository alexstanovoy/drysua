use super::*;

#[derive(Clone, Debug)]
pub(super) struct Estimate {
    pub action: StructuredAction,
    pub score: f64,
    pub metrics: Metrics,
}
#[derive(Debug)]
pub(super) struct Ranking {
    pub estimates: Vec<Estimate>,
    pub best: Vec<StructuredAction>,
    pub separation: f64,
}

pub(super) fn replay(prefix: &Prefix) -> Game {
    assert!(prefix.actions.len() <= 60);
    let mut restored = game(prefix.physics);
    for action in &prefix.actions {
        assert!(!restored.terminal);
        restored.action(*action);
    }
    restored
}

pub(super) fn candidates(game: &Game, space: &ActionSpace) -> Vec<StructuredAction> {
    let mut actions = vec![StructuredAction::Continue];
    for (index, unit) in space.entity_candidates().iter().enumerate() {
        if unit.relation == crate::EntityRelation::Enemy
            && matches!(unit.kind, UnitKind::Hero | UnitKind::CreepMelee)
        {
            actions.push(StructuredAction::AttackUnit {
                unit: ControlledUnit::Hero,
                target: EntityIndex(index),
            });
        }
    }
    actions.extend((0..3).map(cast));
    let tracker = &game.seats[game.side].tracker;
    if tracker.own_hero().is_some_and(|hero| {
        hero.items
            .first()
            .copied()
            .flatten()
            .is_some_and(|item| item.id == ItemId(42))
    }) {
        actions.push(use_mango());
    }
    if tracker
        .own_player()
        .unwrap()
        .stash
        .as_ref()
        .unwrap()
        .iter()
        .flatten()
        .any(|item| item.id == ItemId(42))
    {
        actions.push(StructuredAction::Cast {
            unit: ControlledUnit::Courier,
            slot: AbilitySlot(0),
            target: ActionTarget::None,
        });
    }
    if tracker.own_hero().is_some_and(|hero| hero.mana < 75) {
        let item = space
            .shop_candidates()
            .iter()
            .position(|item| item.item == ItemId(42))
            .unwrap();
        actions.push(StructuredAction::Buy {
            unit: ControlledUnit::Hero,
            item: ShopIndex(item),
        });
    }
    if let Some(action) = recovery_candidate(tracker, space) {
        actions.push(action);
    }
    actions.retain(|action| space.allows(*action));
    assert!(actions.len() <= 10);
    actions
}

fn recovery_candidate(tracker: &StateTracker, space: &ActionSpace) -> Option<StructuredAction> {
    if let Some(hero) = tracker.own_hero().filter(|hero| hero.hp * 3 < hero.max_hp) {
        let fountain = space
            .entity_candidates()
            .iter()
            .find(|unit| {
                unit.kind == UnitKind::Fountain && unit.relation == crate::EntityRelation::Allied
            })
            .map(|unit| unit.position);
        if let Some(fountain) = fountain {
            let point = space
                .point_candidates()
                .iter()
                .enumerate()
                .filter(|(index, _)| space.move_point_mask(ControlledUnit::Hero)[*index])
                .min_by_key(|(_, point)| point.position.distance_squared(fountain))
                .map(|(index, _)| index);
            if let Some(point) = point {
                return Some(StructuredAction::MovePoint {
                    unit: ControlledUnit::Hero,
                    point: PointIndex(point),
                });
            }
        }
        assert!(hero.hp > 0);
    }
    None
}

pub(super) fn estimate(prefix: &Prefix, action: StructuredAction) -> Estimate {
    let mut branch = replay(prefix);
    let baseline = branch.total.score;
    branch.action(action);
    for _ in 1..HORIZON / 3 {
        if branch.terminal {
            break;
        }
        branch.teacher_continuation();
    }
    Estimate {
        action,
        score: branch.total.score - baseline,
        metrics: branch.total,
    }
}

pub(super) fn rank(prefix: &Prefix) -> Ranking {
    let mut current = replay(prefix);
    let (_, space) = current.prepare();
    let estimates: Vec<_> = candidates(&current, &space)
        .into_iter()
        .map(|action| estimate(prefix, action))
        .collect();
    let value = estimates
        .iter()
        .map(|row| row.score)
        .fold(f64::NEG_INFINITY, f64::max);
    let best: Vec<_> = estimates
        .iter()
        .filter(|row| value - row.score <= 1e-6)
        .map(|row| row.action)
        .collect();
    let next = estimates
        .iter()
        .filter(|row| value - row.score > 1e-6)
        .map(|row| row.score)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(!best.is_empty());
    assert!(value.is_finite());
    Ranking {
        estimates,
        best,
        separation: if next.is_finite() { value - next } else { 0.0 },
    }
}
