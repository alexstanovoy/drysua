use super::class_weights;

/// 90 moves, 9 attacks, 1 buy and an unlabeled row.
fn classes() -> Vec<Option<u8>> {
    let mut classes = vec![Some(2); 90];
    classes.extend([Some(16); 9]);
    classes.push(Some(12));
    classes.push(None);
    classes
}

fn class_total(weights: &[f32], class: u8) -> f32 {
    classes()
        .iter()
        .zip(weights)
        .filter(|(row, _)| **row == Some(class))
        .map(|(_, weight)| weight)
        .sum()
}

#[test]
fn class_weights_keep_the_mean_and_shift_weight_to_rare_classes() {
    let flat = class_weights(&classes(), 0.0);
    assert!(flat[..100].iter().all(|weight| *weight == 1.0));
    assert_eq!(flat[100], 0.0);
    // Balanced: every present class carries a third of the 100 labels, but the lone buy
    // hits the cap.
    let balanced = class_weights(&classes(), 1.0);
    assert!((class_total(&balanced, 2) - class_total(&balanced, 16)).abs() < 1e-3);
    assert_eq!(class_total(&balanced, 12), 30.0);
    let half = class_weights(&classes(), 0.5);
    let total: f32 = half.iter().sum();
    assert!((total - 100.0).abs() < 1e-3, "{total}");
    assert!(half[99] > half[90] && half[90] > half[0]);
    assert_eq!(class_weights(&[None, None], 0.5), [0.0, 0.0]);
}
