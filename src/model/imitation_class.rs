//! Imitation classes: the shadow label's action kind, with unit attacks split by
//! target, so rare decisions (buys, consumables, last hits and denies) can weigh
//! as much in the imitation term as the many moves around them.

use crate::{ActionHeadTargets, ActionKind, FeatureFrame, unit_feature};

/// Class names in index order: the sixteen action kinds, where an attack on a
/// hero keeps the `AttackUnit` slot, then attacks on enemy creeps, denies of
/// allied units and every other attack target.
pub const IMITATION_CLASSES: [&str; 19] = [
    "continue",
    "stop",
    "move_point",
    "follow_unit",
    "hold",
    "attack_move_point",
    "attack_hero",
    "cast",
    "use",
    "put_point",
    "put_unit",
    "take",
    "buy",
    "sell",
    "swap",
    "learn",
    "attack_creep",
    "deny",
    "attack_other",
];
const ATTACK_CREEP: usize = ActionKind::COUNT;
const DENY: usize = ActionKind::COUNT + 1;
const ATTACK_OTHER: usize = ActionKind::COUNT + 2;
/// Unit kind one-hot slots: hero, then the four lane creep kinds.
const HERO_KIND: usize = 0;
const LANE_CREEP_KINDS: std::ops::RangeInclusive<usize> = 1..=4;
/// Relation one-hot slots: own, allied, enemy, neutral.
const ALLIED: usize = 1;
const ENEMY: usize = 2;
/// Largest per-row class weight after normalization; bounds one rare row's pull.
const MAX_CLASS_WEIGHT: f32 = 30.0;

/// The class of one label, reading an attack's target from the frame's unit token.
pub(super) fn imitation_class(frame: &FeatureFrame, label: &ActionHeadTargets) -> usize {
    let kind = label.kind.selected;
    assert!(label.kind.active && kind < ActionKind::COUNT);
    if kind != ActionKind::AttackUnit as usize {
        return kind;
    }
    let token = &frame.units[label.entity_pointer.selected];
    let hot = |start: usize, width: usize| (0..width).find(|&slot| token[start + slot] > 0.5);
    let relation = hot(unit_feature::RELATION_START, 4);
    match (hot(unit_feature::KIND_START, 12), relation) {
        (Some(HERO_KIND), Some(ENEMY)) => kind,
        (Some(unit), Some(ENEMY)) if LANE_CREEP_KINDS.contains(&unit) => ATTACK_CREEP,
        (Some(unit), Some(ALLIED)) if LANE_CREEP_KINDS.contains(&unit) => DENY,
        _ => ATTACK_OTHER,
    }
}

/// Per-row imitation weights over one update: labeled rows of a class that holds
/// `n` of the `N` labels over `K` present classes weigh `(N / (K n))^balance`,
/// normalized to mean one over labeled rows and capped; unlabeled rows weigh zero.
/// `balance` 0 weighs every label alike, 1 gives every present class equal weight.
#[allow(
    clippy::float_arithmetic,
    clippy::cast_precision_loss,
    reason = "bounded loss weights from row counts"
)]
pub(super) fn class_weights(classes: &[Option<u8>], balance: f32) -> Vec<f32> {
    assert!((0.0..=1.0).contains(&balance));
    let mut counts = [0usize; IMITATION_CLASSES.len()];
    for class in classes.iter().flatten() {
        counts[usize::from(*class)] += 1;
    }
    let labeled: usize = counts.iter().sum();
    let present = counts.iter().filter(|&&count| count > 0).count();
    if labeled == 0 {
        return vec![0.0; classes.len()];
    }
    let raw = counts.map(|count| {
        if count == 0 {
            0.0
        } else {
            (labeled as f64 / (present * count) as f64).powf(f64::from(balance))
        }
    });
    let total: f64 = raw
        .iter()
        .zip(counts)
        .map(|(weight, count)| weight * count as f64)
        .sum();
    let scale = labeled as f64 / total;
    classes
        .iter()
        .map(|class| {
            class.map_or(0.0, |class| {
                ((raw[usize::from(class)] * scale) as f32).min(MAX_CLASS_WEIGHT)
            })
        })
        .collect()
}

#[cfg(test)]
#[path = "../tests/imitation_class.rs"]
mod tests;
