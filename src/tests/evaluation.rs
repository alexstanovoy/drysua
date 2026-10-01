use std::path::Path;

use super::super::eval_players::{Role, load_player};
use super::*;

const FIRST_SEED: u64 = 11;
const CANDIDATE_SEED: u64 = 5;
const OPPONENT_SEED: u64 = 6;

fn fresh_weights(seed: u64) -> crate::TestDirectory {
    let directory = crate::test_directory("evaluation-weights");
    let model = PolicyModel::fresh(seed).expect("model");
    crate::TrainingArtifact::save_runtime_weights(&model, &directory).expect("runtime weights");
    directory
}

fn entry(name: &str, player: PlayerSpec, role: Role) -> PoolEntry {
    PoolEntry {
        name: name.to_owned(),
        player,
        role,
    }
}

/// A neural candidate against a rule opponent and a neural opponent.
fn settings(
    candidate: &Path,
    opponent: &Path,
    parallel: usize,
    groups: usize,
) -> EvaluationSettings {
    EvaluationSettings {
        candidate_name: "fresh".to_owned(),
        candidate: PlayerSpec::Weights(candidate.to_path_buf()),
        pool: vec![
            entry(
                "teacher",
                PlayerSpec::Script(ScriptKind::Teacher),
                Role::Train,
            ),
            entry(
                "other",
                PlayerSpec::Weights(opponent.to_path_buf()),
                Role::HeldOut,
            ),
        ],
        first_seed: FIRST_SEED,
        seeds: 1,
        parallel,
        groups,
        greedy: false,
        device: PolicyDevice::Cpu,
    }
}

fn play(parallel: usize, groups: usize) -> Vec<(PlannedGame, GameSummary)> {
    let (candidate, opponent) = (fresh_weights(CANDIDATE_SEED), fresh_weights(OPPONENT_SEED));
    let settings = settings(&candidate, &opponent, parallel, groups);
    let load = |spec| load_player(spec, settings.device).expect("player");
    let models = Models {
        candidate: load(&settings.candidate),
        pool: settings
            .pool
            .iter()
            .map(|entry| load(&entry.player))
            .collect(),
    };
    let mut games = play_all(&settings, &models).expect("games");
    games.sort_by_key(|(game, _)| *game);
    games
}

/// Full games are the expensive part, so the contract tests share one batched run
/// whose single inference batch mixes both models' rows.
fn batched_games() -> &'static [(PlannedGame, GameSummary)] {
    static GAMES: std::sync::OnceLock<Vec<(PlannedGame, GameSummary)>> = std::sync::OnceLock::new();
    GAMES.get_or_init(|| play(4, 1))
}

#[test]
fn pool_games_depend_on_players_and_seeds_but_not_batching() {
    let pipelined = play(1, 2);
    assert_eq!(batched_games(), pipelined.as_slice());
}

#[test]
fn pool_evaluation_plays_each_seed_once_per_side_and_opponent() {
    let games: Vec<(usize, u64, &str)> = batched_games()
        .iter()
        .map(|(game, summary)| (game.opponent, game.seed, summary.side_label()))
        .collect();
    assert_eq!(games.len(), 4);
    for opponent in 0..2 {
        let sides: Vec<&str> = games
            .iter()
            .filter(|game| game.0 == opponent && game.1 == FIRST_SEED)
            .map(|game| game.2)
            .collect();
        assert_eq!(sides.len(), 2, "opponent {opponent}");
        assert_ne!(sides[0], sides[1], "opponent {opponent}");
    }
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
                    super::super::game_summary::EndReason::Draw
                        | super::super::game_summary::EndReason::TimeCap
                ));
                continue;
            }
        };
        match summary.end_reason {
            super::super::game_summary::EndReason::Tower => {
                assert_eq!(loser.tower_hp, 0.0, "{game:?}");
            }
            super::super::game_summary::EndReason::Deaths => {
                assert!(loser.deaths > winner.deaths, "{game:?}");
            }
            other => panic!("decided game ended by {other:?}"),
        }
    }
}

/// Teacher last-hits and buys; every game tick after the pregame places each hero once.
#[test]
fn evaluation_economy_sees_teacher_farm_and_places_every_tick() {
    let mut teacher_last_hits = 0;
    for (game, summary) in batched_games() {
        let json = summary.json();
        for hero in ["own", "enemy"] {
            let places = json[hero]["spending"]["places"]
                .as_object()
                .expect("places");
            let placed: u64 = places
                .values()
                .map(|ticks| ticks.as_u64().expect("ticks"))
                .sum();
            assert_eq!(
                placed,
                u64::from(summary.ticks - crate::MAP2_PREGAME_TICKS + 1),
                "{game:?}"
            );
            assert!(
                json[hero]["economy"]["net_worth"]
                    .as_i64()
                    .expect("net worth")
                    > 0
            );
        }
        if game.opponent == 0 {
            let teacher = &json["enemy"];
            teacher_last_hits += teacher["economy"]["last_hits"].as_u64().expect("last hits");
            assert!(
                !teacher["spending"]["items_bought"]
                    .as_object()
                    .expect("items")
                    .is_empty()
            );
        }
    }
    assert!(teacher_last_hits > 0);
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
        let mut leads = 0;
        for key in ["side", "outcome", "end_reason"] {
            assert_eq!(fields[key], json[key], "{key}");
        }
        assert_eq!(fields["ticks"], json["ticks"].to_string());
        for hero in ["own", "enemy"] {
            let expected = &json[hero];
            for key in [
                "kills",
                "deaths",
                "level",
                "xp",
                "raze_hero_hits",
                "raze_hits",
            ] {
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
            for (mode, count) in expected["raze_modes"].as_object().expect("raze modes") {
                let field = format!("{hero}_raze_mode_{mode}");
                assert_eq!(fields[field.as_str()], count.to_string(), "{field}");
            }
            for (key, value) in expected["economy"].as_object().expect("economy") {
                let field = format!("{hero}_{key}");
                assert_eq!(fields[field.as_str()], value.to_string(), "{field}");
            }
            for (place, ticks) in expected["spending"]["places"].as_object().expect("places") {
                let field = format!("{hero}_ticks_{place}");
                assert_eq!(fields[field.as_str()], ticks.to_string(), "{field}");
            }
            for (minute, standing) in expected["minutes"].as_object().expect("minutes") {
                if standing.is_null() {
                    continue;
                }
                for key in ["net_worth", "last_hits"] {
                    let field = format!("{hero}_{key}_{minute}");
                    assert_eq!(fields[field.as_str()], standing[key].to_string(), "{field}");
                    leads += 1;
                }
            }
        }
        for (minute, lead) in json["leads"].as_object().expect("leads") {
            let Some(lead) = lead.as_object() else {
                continue;
            };
            for (key, value) in lead {
                let field = format!("lead_{minute}_{key}");
                assert_eq!(fields[field.as_str()], value.to_string(), "{field}");
                leads += 1;
            }
        }
        assert!(leads > 0, "every game lasts past the first milestone");
        assert_eq!(fields.len(), 4 + 2 * (14 + 6 + 5) + leads);
    }
}

/// The header and game lines are the contract `scripts/eval_pool.py` reads.
#[test]
fn evaluation_writes_a_header_then_every_game_in_plan_order() {
    let directory = crate::test_directory("evaluation-output");
    let output = directory.join("result.jsonl");
    let settings = EvaluationSettings {
        candidate_name: "teacher".to_owned(),
        candidate: PlayerSpec::Script(ScriptKind::Teacher),
        pool: vec![entry(
            "harass",
            PlayerSpec::Script(ScriptKind::HarassPush),
            Role::HeldOut,
        )],
        first_seed: FIRST_SEED,
        seeds: 1,
        parallel: 2,
        groups: 1,
        greedy: false,
        device: PolicyDevice::Cpu,
    };
    run_evaluation(&settings, &output).expect("evaluation");
    let text = std::fs::read_to_string(&output).expect("output");
    let lines: Vec<Value> = text
        .lines()
        .map(|line| serde_json::from_str(line).expect("json line"))
        .collect();
    assert_eq!(lines.len(), 3);
    let header = &lines[0];
    assert_eq!(header["kind"], "header");
    assert_eq!(header["candidate"]["key"], "script:teacher");
    assert_eq!(header["pool"][0]["name"], "harass");
    assert_eq!(header["pool"][0]["role"], "held-out");
    assert_eq!(header["pool"][0]["key"], "script:harass-push");
    assert_eq!(header["seeds"]["count"], 1);
    let context = header["context"].as_str().expect("context");
    assert!(context.starts_with("exe:") && context.ends_with("/sampled"));
    for (line, side) in lines[1..].iter().zip(["radiant", "dire"]) {
        assert_eq!(line["kind"], "game");
        assert_eq!(line["opponent"], "harass");
        assert_eq!(line["seed"], FIRST_SEED);
        assert_eq!(line["side"], side);
    }
}

#[test]
fn evaluation_never_replaces_an_existing_result() {
    let directory = crate::test_directory("evaluation-existing");
    let output = directory.join("result.jsonl");
    std::fs::write(&output, b"earlier").expect("earlier result");
    let missing = Path::new("/nonexistent");
    let error =
        run_evaluation(&settings(missing, missing, 1, 1), &output).expect_err("existing output");
    assert!(
        error
            .to_string()
            .contains("already exists; evaluation never replaces a result")
    );
    assert_eq!(std::fs::read(&output).expect("kept"), b"earlier");
}

/// An average candidate is the element-wise mean of its members, in parameter order.
#[test]
#[allow(clippy::float_arithmetic, reason = "expected parameter means")]
fn average_player_is_the_parameter_mean_of_its_members() {
    let first = fresh_weights(CANDIDATE_SEED);
    let second = fresh_weights(OPPONENT_SEED);
    let parameters = |spec: &PlayerSpec| {
        load_player(spec, PolicyDevice::Cpu)
            .expect("player")
            .model()
            .expect("neural")
            .export_parameters()
            .expect("parameters")
    };
    let a = parameters(&PlayerSpec::Weights(first.to_path_buf()));
    let b = parameters(&PlayerSpec::Weights(second.to_path_buf()));
    let mean = parameters(&PlayerSpec::Average(vec![
        first.to_path_buf(),
        second.to_path_buf(),
    ]));
    assert_ne!(a, b);
    for ((a, b), mean) in a.iter().zip(&b).zip(&mean) {
        let expected = ((f64::from(*a) + f64::from(*b)) / 2.0) as f32;
        assert_eq!(expected.to_bits(), mean.to_bits());
    }
}
