use super::*;

const FIRST_SEED: u64 = 11;
const WEIGHTS_SEED: u64 = 5;

fn fresh_weights() -> PathBuf {
    let directory = crate::test_directory("evaluation-weights");
    let model = PolicyModel::fresh(WEIGHTS_SEED).expect("model");
    TrainingArtifact::save_runtime_weights(&model, &directory).expect("runtime weights");
    directory
}

fn settings(candidate: PathBuf, parallel: usize, groups: usize) -> EvaluationSettings {
    EvaluationSettings {
        candidate,
        opponent: EvaluationOpponent::Teacher,
        first_seed: FIRST_SEED,
        seeds: 1,
        parallel,
        groups,
        greedy: false,
        device: PolicyDevice::Cpu,
    }
}

fn play(parallel: usize, groups: usize) -> Vec<(PlannedGame, GameSummary)> {
    let settings = settings(fresh_weights(), parallel, groups);
    let models = Models {
        learner: load_model(&settings.candidate, settings.device).expect("model"),
        opponent: None,
    };
    let mut games = play_all(&settings, &models).expect("games");
    games.sort_by_key(|(game, _)| (game.seed, game.seat));
    games
}

/// Full games are the expensive part, so the contract tests share one batched run.
fn batched_games() -> &'static [(PlannedGame, GameSummary)] {
    static GAMES: std::sync::OnceLock<Vec<(PlannedGame, GameSummary)>> = std::sync::OnceLock::new();
    GAMES.get_or_init(|| play(2, 1))
}

#[test]
fn evaluation_games_depend_on_weights_and_seeds_but_not_batching() {
    let pipelined = play(1, 2);
    assert_eq!(batched_games(), pipelined.as_slice());
}

#[test]
fn evaluation_plays_each_seed_once_per_side() {
    let sides: Vec<(u64, &str)> = batched_games()
        .iter()
        .map(|(game, summary)| (game.seed, summary.side_label()))
        .collect();
    assert_eq!(sides.len(), 2);
    assert_eq!(sides[0].0, FIRST_SEED);
    assert_eq!(sides[1].0, FIRST_SEED);
    assert_ne!(sides[0].1, sides[1].1);
}

#[test]
fn evaluation_end_reasons_agree_with_final_state() {
    for (game, summary) in batched_games() {
        let (winner, loser) = match summary.outcome {
            Some(PpoTerminalOutcome::Win) => (summary.own, summary.enemy),
            Some(PpoTerminalOutcome::Loss) => (summary.enemy, summary.own),
            _ => {
                assert!(matches!(
                    summary.end_reason,
                    EndReason::Draw | EndReason::TimeCap
                ));
                continue;
            }
        };
        match summary.end_reason {
            EndReason::Tower => assert_eq!(loser.tower_hp, 0.0, "seat {}", game.seat),
            EndReason::Deaths => assert!(loser.deaths > winner.deaths, "seat {}", game.seat),
            other => panic!("decided game ended by {other:?}"),
        }
    }
}

/// The dashboard reads the `key=value` log line; it must carry every JSON field.
#[test]
fn episode_summary_log_carries_every_evaluation_field() {
    for (_, summary) in batched_games() {
        let log = summary.to_string();
        let fields: std::collections::BTreeMap<&str, &str> = log
            .split(' ')
            .map(|field| field.split_once('=').expect("key=value"))
            .collect();
        let json = summary.json();
        for key in ["side", "outcome", "end_reason"] {
            assert_eq!(fields[key], json[key], "{key}");
        }
        assert_eq!(fields["ticks"], json["ticks"].to_string());
        for hero in ["own", "enemy"] {
            let expected = &json[hero];
            for key in ["kills", "deaths", "level", "xp", "raze_hero_hits"] {
                let field = format!("{hero}_{key}");
                assert_eq!(fields[field.as_str()], expected[key].to_string(), "{field}");
            }
            let tower: f64 = fields[format!("{hero}_tower_hp").as_str()]
                .parse()
                .expect("tower");
            assert_eq!(tower, expected["tower_hp"].as_f64().expect("tower json"));
            for (cast, count) in expected["casts"].as_object().expect("casts") {
                let field = format!("{hero}_casts_{cast}");
                assert_eq!(fields[field.as_str()], count.to_string(), "{field}");
            }
        }
        assert_eq!(fields.len(), 4 + 2 * 10);
    }
}

#[test]
fn evaluation_never_replaces_an_existing_result() {
    let output = crate::test_directory("evaluation-existing").join("result.jsonl");
    std::fs::write(&output, b"earlier").expect("earlier result");
    let error = run_evaluation(&settings(PathBuf::from("/nonexistent"), 1, 1), &output)
        .expect_err("existing output");
    assert!(
        error
            .to_string()
            .contains("already exists; evaluation never replaces a result")
    );
    assert_eq!(std::fs::read(&output).expect("kept"), b"earlier");
}
