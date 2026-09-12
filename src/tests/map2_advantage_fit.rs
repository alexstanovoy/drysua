use super::*;
use crate::ppo_arena::map2_transfer_fix as opening;
use sha2::{Digest, Sha256};

const INITIAL_SHA: &str = "1739d280cb6c3fbd0df71ffe8c4a129ed3e25a0ed294b0b57931c4891c566bf4";
const COVERAGE: [&str; 10] = [
    "duel",
    "empty",
    "combo",
    "finish_two",
    "creep_low",
    "creep_healthy",
    "mango_combat",
    "supply",
    "recover",
    "finish",
];

struct Row {
    set: CheckedActionSet,
    space: ActionSpace,
    seed: u64,
    side: usize,
    origin: &'static str,
}

pub(super) fn cases(training: bool) -> Vec<(&'static str, Physics)> {
    let mut cases = Vec::new();
    for (index, coverage) in COVERAGE.into_iter().enumerate() {
        for variant in if training { [0, 2] } else { [1, 3] } {
            for side in 0..2 {
                let mut physics = Physics {
                    seed: (if training { 10094000 } else { 10095000 })
                        + index as u64 * 16
                        + variant * 2
                        + side as u64,
                    side,
                    tick: 901 + variant as u32 * 150,
                    level: if variant < 2 { 3 } else { 4 },
                    own_hp: 680,
                    enemy_hp: 680,
                    mana: 353,
                    reach: 450,
                    facing: if side == 0 { 0 } else { 32768 },
                    creep_hp: None,
                    mango: false,
                    stash: false,
                    gold: 0,
                    at_home: false,
                    deaths: 0,
                };
                configure_coverage(&mut physics, coverage, variant);
                cases.push((coverage, physics));
            }
        }
    }
    assert_eq!(cases.len(), 40);
    cases
}

fn configure_coverage(physics: &mut Physics, coverage: &str, variant: u64) {
    match coverage {
        "duel" => {
            physics.reach = [300, 350, 600, 650][variant as usize];
        }
        "empty" => physics.enemy_hp = 0,
        "combo" => {
            physics.enemy_hp = 330;
            physics.own_hp = 120;
            physics.deaths = 1;
        }
        "finish_two" => {
            physics.enemy_hp = 130;
            physics.own_hp = 200;
            physics.deaths = 1;
        }
        "creep_low" | "creep_healthy" => {
            physics.enemy_hp = 0;
            physics.creep_hp = Some(if coverage == "creep_low" { 35 } else { 550 });
            physics.facing = physics.facing.wrapping_add(variant as u16 * 2048);
        }
        "mango_combat" => {
            physics.enemy_hp = 60;
            physics.own_hp = 80;
            physics.mana = 0;
            physics.mango = true;
            physics.deaths = 1;
        }
        "supply" => {
            physics.enemy_hp = 60;
            physics.own_hp = 100;
            physics.mana = 0;
            physics.gold = 65;
            physics.at_home = true;
            physics.deaths = 1;
        }
        "recover" | "finish" => {
            physics.own_hp = 100;
            physics.enemy_hp = if coverage == "finish" { 30 } else { 680 };
            physics.deaths = 1;
        }
        _ => unreachable!(),
    }
}

fn collect_returns(training: bool, report: &mut String) -> Vec<Row> {
    let mut rows = Vec::new();
    for (coverage, physics) in cases(training) {
        let mut prefix = Prefix {
            physics,
            actions: Vec::new(),
        };
        for decision in 0..8 {
            let mut state = rank::replay(&prefix);
            if state.terminal {
                break;
            }
            let (frame, space) = state.prepare();
            let ranking = rank::rank(&prefix);
            let canonical = ranking.best[0];
            let retained = ranking.best.len() <= 4 && ranking.separation >= 1e-4;
            if retained {
                let set = CheckedActionSet::new(frame, &space, &ranking.best).unwrap();
                rows.push(Row {
                    set,
                    space,
                    seed: physics.seed,
                    side: physics.side,
                    origin: "counterfactual_observed_return",
                });
            }
            writeln!(report, "coverage={coverage} seed={} side={} decision={decision} retained={retained} best={:?} separation={} scores={:?}", physics.seed, physics.side, ranking.best, ranking.separation, ranking.estimates.iter().map(|row| (row.action, row.score)).collect::<Vec<_>>()).unwrap();
            prefix.actions.push(canonical);
            if !retained
                || !ranking
                    .estimates
                    .iter()
                    .any(|row| row.action.kind() == ActionKind::AttackUnit)
            {
                break;
            }
        }
    }
    assert!(rows.len() <= 1024);
    assert!(!rows.is_empty());
    rows
}

fn anchors(training: bool) -> Vec<Row> {
    let base = if training { 10094980 } else { 10095980 };
    [base, base + 1]
        .into_iter()
        .flat_map(|seed| (0..2).map(move |side| (seed, side)))
        .flat_map(|(seed, side)| opening::data::collect_anchor(seed, side))
        .map(|row| Row {
            set: CheckedActionSet::new(row.sample.frame().clone(), &row.space, &[row.action])
                .unwrap(),
            space: row.space,
            seed: row.seed,
            side: row.side,
            origin: "unchanged_teacher_opening_rehearsal",
        })
        .collect()
}

#[test]
#[ignore = "One guarded corrected-target fit from initial, with common-policy observed-return forks."]
fn corrected_advantage_fit() {
    let started = Instant::now();
    let mut report = String::from(
        "labels=counterfactual_observed_return\ncontinuation=unchanged_teacher_after3ticks\nopponent=unchanged_teacher\nhorizon=90\nreward=complete_seat_Map2Reward_no_horizon_terminal\n",
    );
    let skills = collect_returns(true, &mut report);
    let anchors = anchors(true);
    assert!(skills.len() + anchors.len() <= 4096);
    let source = initial_path();
    assert_eq!(
        file_hash(&source.join("drysua.weights.safetensors")),
        INITIAL_SHA
    );
    let model = load(&source);
    writeln!(
        report,
        "training_rows={} anchors={} data_sha={} anchor_sha={} construction_seconds={:.3}",
        skills.len(),
        anchors.len(),
        row_hash(&skills),
        row_hash(&anchors),
        started.elapsed().as_secs_f64()
    )
    .unwrap();
    let steps = fit(&model, &skills, &anchors, started, &mut report);
    writeln!(report, "optimizer_steps={steps}\nsource_sha256={INITIAL_SHA}\ndiagnostic_only=true\nqualified=false\nwall_seconds={:.3}", started.elapsed().as_secs_f64()).unwrap();
    write_new(&root().join("FIT.txt"), &report);
    assert_eq!(steps, 512, "bounded fit incomplete: no export and no retry");
    let target = root().join("weights");
    assert!(!target.exists());
    std::fs::create_dir(&target).unwrap();
    TrainingArtifact::save_runtime_weights(&model, &target).unwrap();
    let hash = file_hash(&target.join("drysua.weights.safetensors"));
    write_new(
        &target.join("MANIFEST.txt"),
        &format!(
            "diagnostic_only=true\nqualified=false\nweights_sha256={hash}\ninitial_sha256={INITIAL_SHA}\nmodel=16\nfeature=14\noptimizer=fresh_Adam_lr1e-4_beta.9_.999_eps1e-8_clip.5\nupdates=512\nbatch_states=16\nrehearsal_fraction=0.5\nsampling_seed=10094999\ntraining_data_sha={}\nanchor_data_sha={}\n",
            row_hash(&skills),
            row_hash(&anchors)
        ),
    );
    eprintln!(
        "rows={} anchors={} steps={steps} seconds={:.3} weights_sha256={hash}",
        skills.len(),
        anchors.len(),
        started.elapsed().as_secs_f64()
    );
}

fn fit(
    model: &PolicyModel,
    skills: &[Row],
    anchors: &[Row],
    started: Instant,
    report: &mut String,
) -> u64 {
    let mut optimizer = model
        .claim_optimizer(AdamConfig {
            learning_rate: 0.0001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            gradient_clip: 0.5,
        })
        .unwrap();
    let mut random = PpoRng::new(10094999);
    let fit_started = Instant::now();
    for step in 1..=512 {
        if started.elapsed() > Duration::from_secs(250) {
            break;
        }
        let mut batch = Vec::with_capacity(16);
        for _ in 0..8 {
            batch.push(&anchors[random.below(anchors.len() as u64).unwrap() as usize].set);
        }
        for _ in 0..8 {
            batch.push(&skills[random.below(skills.len() as u64).unwrap() as usize].set);
        }
        let update = model
            .train_checked_action_sets(&batch, &mut optimizer)
            .unwrap();
        if step == 1 || step % 64 == 0 {
            writeln!(
                report,
                "step={step} loss={} fit_seconds={:.3}",
                update.average_loss,
                fit_started.elapsed().as_secs_f64()
            )
            .unwrap();
            eprintln!(
                "step={step} loss={} fit_seconds={:.3}",
                update.average_loss,
                fit_started.elapsed().as_secs_f64()
            );
        }
    }
    optimizer.step()
}

fn closed(model: &PolicyModel, physics: Physics) -> (Metrics, bool) {
    let mut state = game(physics);
    let mut prefix = Prefix {
        physics,
        actions: Vec::new(),
    };
    let mut beneficial_third = false;
    for _ in 0..60 {
        if state.terminal {
            break;
        }
        let (frame, space) = state.prepare();
        let action = model.choose(&frame, &space).unwrap().action;
        if let StructuredAction::Cast {
            unit: ControlledUnit::Hero,
            slot,
            ..
        } = action
            && slot.0 < 3
            && state.total.hit_slots.count_ones() == 2
            && state.total.hit_slots & (1 << slot.0) == 0
        {
            let chosen = rank::estimate(&prefix, action);
            let alternative = rank::candidates(&state, &space)
                .into_iter()
                .filter(|action| action.kind() != ActionKind::Cast)
                .map(|action| rank::estimate(&prefix, action).score)
                .fold(f64::NEG_INFINITY, f64::max);
            beneficial_third |= chosen.score > alternative + 1e-4;
        }
        state.action(action);
        prefix.actions.push(action);
    }
    (state.total, beneficial_third)
}

#[test]
#[ignore = "Prospective return/opening gate of the single fixed model; old label agreement cannot promote it."]
fn prospective_outcome_gate() {
    let started = Instant::now();
    let model = load(&root().join("weights"));
    let initial = load(&initial_path());
    let mut report = String::new();
    let mut delta = [0.0; 2];
    let mut non_worse = [0; 2];
    let mut empty = true;
    let mut mango = [0; 2];
    let mut combo = [0; 2];
    for (coverage, physics) in cases(false) {
        let (before, _) = closed(&initial, physics);
        let (after, beneficial_third) = closed(&model, physics);
        let gain = after.score - before.score;
        delta[physics.side] += gain;
        non_worse[physics.side] += u32::from(gain >= -1e-4);
        if coverage == "empty" {
            empty &= after.casts == 0 && after.mana == 0;
        }
        if coverage == "mango_combat" {
            let prefix = Prefix {
                physics,
                actions: Vec::new(),
            };
            let used = rank::estimate(&prefix, use_mango());
            let delayed = rank::estimate(&prefix, StructuredAction::Continue);
            let benefit = used.score - delayed.score;
            mango[physics.side] += u32::from(after.mango > 0 && benefit > 1e-4);
            writeln!(
                report,
                "mango_counterfactual seed={} use_minus_continue={benefit}",
                physics.seed
            )
            .unwrap();
        }
        if after.hit_slots.count_ones() == 3 && beneficial_third {
            combo[physics.side] += 1;
        }
        writeln!(report, "coverage={coverage} physics={physics:?} initial={before:?} learned={after:?} gain={gain} beneficial_third={beneficial_third}").unwrap();
    }
    let mut opening_pass = true;
    for seed in [10095980, 10095981] {
        for side in 0..2 {
            let observed = opening::run_opening(&model, Some(&initial), seed, side);
            let pass = opening::progress_gate(&observed.stats);
            opening_pass &= pass;
            writeln!(
                report,
                "opening seed={seed} side={side} pass={pass} {:?}",
                observed.stats
            )
            .unwrap();
        }
    }
    let pass = opening_pass
        && empty
        && (0..2).all(|side| {
            delta[side] / 20.0 >= 0.001
                && non_worse[side] >= 15
                && mango[side] > 0
                && combo[side] > 0
        });
    writeln!(report, "mean_gain={:?}\nnon_worse={non_worse:?}/20\nempty_no_waste={empty}\nbeneficial_mango={mango:?}\nbeneficial_three_hits={combo:?}\nopening_pass={opening_pass}\noutcome_gate={pass}\ndev_permitted={pass}\nqualified=false\nwall_seconds={:.3}", delta.map(|value| value / 20.0), started.elapsed().as_secs_f64()).unwrap();
    write_new(&root().join("GATES.txt"), &report);
    eprintln!("{report}");
}

fn row_hash(rows: &[Row]) -> String {
    let mut digest = Sha256::new();
    for row in rows {
        digest.update(format!(
            "{}:{}:{}:{}:{:?}",
            row.origin,
            row.seed,
            row.side,
            row.space.tick(),
            row.set.actions()
        ));
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
            digest.update(value.to_bits().to_le_bytes());
        }
    }
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
fn initial_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/temp/map2-learning-20260911/initial")
}
fn load(path: &Path) -> PolicyModel {
    let model = PolicyModel::fresh_on(10094999, PolicyDevice::Cpu).unwrap();
    TrainingArtifact::load_runtime_weights(&model, path).unwrap();
    model
}
fn file_hash(path: &Path) -> String {
    Sha256::digest(std::fs::read(path).unwrap())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
fn write_new(path: &Path, text: &str) {
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
