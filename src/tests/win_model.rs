//! The learned potential's fit: it must rank leads correctly, respect the seat
//! swap and be reproducible bit for bit.

use super::*;

/// Games whose win chance rises with the XP lead, one sample each.
fn window() -> VecDeque<WinGame> {
    (0..400u64)
        .map(|index| {
            let lead = (index % 21) as f32 / 10.0 - 1.0;
            let mut sample = [0.0; BASE];
            sample[0] = 0.3;
            sample[1] = lead;
            let chance = 50.0 + 40.0 * f64::from(lead);
            let win = ((index.wrapping_mul(7_919) % 100) as f64) < chance;
            WinGame {
                samples: vec![sample],
                score: if win { 2 } else { 0 },
            }
        })
        .collect()
}

#[test]
fn fit_ranks_leads_is_seat_symmetric_and_reproducible() {
    let games = window();
    let model = fit(&games, 7);
    let at = |lead: f32| {
        let mut sample = [0.0; BASE];
        sample[0] = 0.3;
        sample[1] = lead;
        sample
    };
    assert!(model.probability(&at(1.0)) > 0.7);
    assert!(model.probability(&at(-1.0)) < 0.3);
    let ahead = at(0.6);
    let swapped = model.probability(&ahead) + model.probability(&mirror(&ahead));
    assert!((swapped - 1.0).abs() < 1.0e-6, "{swapped}");
    assert_eq!(fit(&games, 7), model);
    let pairs: Vec<(f64, bool)> = games
        .iter()
        .map(|game| (model.probability(&game.samples[0]), game.score == 2))
        .collect();
    assert!(auc(&pairs) > 0.7);
    assert_eq!(auc(&[(0.2, false), (0.2, true)]), 0.5);
}
