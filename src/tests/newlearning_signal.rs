//! Bounded frozen-policy census; no optimizer or readout fitting.
#![allow(clippy::float_arithmetic, reason = "Diagnostic return reconstruction")]

use super::super::{TrainingEnvironment, build_environment, setup_seats};
use super::*;
use crate::{ActionKind, ControlledUnit, PpoBatch, PpoPolicyChoice, StructuredAction};
use std::{collections::BTreeMap, hash::Hasher};

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
#[ignore = "explicit U428 checkpoint and exclusive bounded CUDA runner required"]
fn learning_signal_probe_cuda() {
    std::thread::Builder::new()
        .name("learning-signal".into())
        .stack_size(8 * 1024 * 1024)
        .spawn(probe_frozen)
        .unwrap()
        .join()
        .unwrap();
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
fn probe_frozen() {
    let _flush = FlushPerformanceLogs;
    assert!(!prometheus::enabled());
    let directory = PathBuf::from(
        std::env::var_os("DRYSUA_LEARNING_CHECKPOINT")
            .filter(|value| !value.is_empty())
            .expect("explicit U428 directory"),
    );
    let artifact = TrainingArtifact::load(&directory).expect("strict U428 artifact");
    assert_eq!(artifact.progress().global_update, 428);
    assert_eq!(artifact.run().run_seed, 9001);
    let config = artifact.config();
    assert_eq!(config.environments, 40);
    assert_eq!((config.gamma_tick, config.gae_lambda), (1.0, 0.98));
    let model = PolicyModel::fresh_on(1, PolicyDevice::Cuda { ordinal: 0 }).unwrap();
    TrainingArtifact::load_runtime_weights(&model, &directory).unwrap();
    let before = model.export_parameters().unwrap();
    let identity = model.policy_identity().unwrap();
    let actor = artifact
        .progress()
        .rng_states
        .iter()
        .find(|state| state.name() == "ppo_actor_sampling")
        .unwrap();
    assert_eq!(actor.draws(), 17120);
    let mut master = PpoRng::from_checkpoint(actor.state(), actor.draws()).unwrap();
    let seats = balanced_policy_seats(9001, 428, 40).unwrap();
    let environments = seats
        .iter()
        .enumerate()
        .map(|(index, seat)| {
            let game = 17120 + index as u64;
            build_environment(
                derive_training_seed(9001, game, crate::randomization::ARENA_DOMAIN),
                derive_training_seed(9001, game, crate::randomization::OPPONENT_DOMAIN),
                MapId(2),
                *seat,
                0,
                OpponentSpec::Teacher,
            )
            .unwrap()
        })
        .collect();
    eprintln!(
        "learning-start block=428 first_game=17120 games=40 clean=true seed=9001 gamma=1 lambda=0.98 advantages=raw coverage=pre_action_visible_disk_proxy engine=patched_composite_buy_courier_errands action_mask=tango_approach historical_trajectory_equivalence=false head_fits=none"
    );
    let result = collect(
        &model,
        config,
        environments,
        actor_stream_rngs(&mut master, 40).unwrap(),
        true,
    );
    log_census(&analyze(&result.batch, &result.tapes, config.gae_lambda));
    crate::tests::support::assert_bits(&model.export_parameters().unwrap(), &before);
    assert_eq!(model.policy_identity().unwrap(), identity);
    let report = result.report;
    eprintln!(
        "learning-finish wins={} losses={} draws={} timeouts={} decisions={} ticks={} samples={} production_optimizer_steps_applied=0 head_fits=0 parameters_unchanged=true parameter_count={} actor_copy_draws={}",
        report.terminal_wins,
        report.terminal_losses,
        report.terminal_draws,
        report.episode_timeouts,
        result.tapes.iter().map(Vec::len).sum::<usize>(),
        report.elapsed_ticks,
        result.batch.len(),
        before.len(),
        master.draws()
    );
}

struct Collection {
    batch: PpoBatch,
    random: Vec<PpoRng>,
    report: PpoSmokeReport,
    tapes: Vec<Vec<Decision>>,
    traces: Vec<u64>,
}

fn collect(
    model: &PolicyModel,
    mut config: PpoConfig,
    mut environments: Vec<TrainingEnvironment>,
    mut random: Vec<PpoRng>,
    enabled: bool,
) -> Collection {
    let count = environments.len();
    assert!((1..=PPO_ANNEALED_MAX_GAMES).contains(&count));
    assert_eq!(config.gamma_tick, 1.0);
    let mut streams = (0..count)
        .map(|index| {
            let mut stream = episode::game_stream(9001, 17120 + index as u64).unwrap();
            stream.learning_signal = enabled.then(|| Vec::with_capacity(ACTOR_DECISIONS));
            stream
        })
        .collect::<Vec<_>>();
    let mut rollout = PpoRollout::for_config(config, model.policy_identity().unwrap()).unwrap();
    let mut report = PpoSmokeReport::default();
    episode::collect_batch(
        model,
        config,
        0,
        "learning-probe",
        &mut environments,
        &mut streams,
        &mut random,
        ACTOR_DECISIONS,
        &mut rollout,
        &mut report,
    )
    .unwrap();
    assert_eq!(report.completed_episodes.ordered_outcomes().len(), count);
    for environment in &environments {
        reject_production_rejection(environment, "learning census").unwrap();
    }
    let traces = streams.iter().map(|stream| stream.trace.finish()).collect();
    let tapes: Vec<Vec<Decision>> = streams
        .into_iter()
        .filter_map(|stream| stream.learning_signal)
        .collect();
    if enabled {
        assert_eq!(tapes.len(), count);
        let total: f64 = tapes.iter().flatten().map(|row| row.reward).sum();
        assert!((total - report.map2_reward.total).abs() < 1e-9);
    }
    config.gae_lambda = 1.0;
    Collection {
        batch: rollout.finish(config).unwrap(),
        random,
        report,
        tapes,
        traces,
    }
}

pub(in crate::ppo_arena) struct Decision {
    tick: u32,
    action: StructuredAction,
    retained: bool,
    coverage: Option<u8>,
    pub(in crate::ppo_arena) reward: f64,
    pub(in crate::ppo_arena) ticks: u32,
    pub(in crate::ppo_arena) terminal: bool,
}
const _: () = assert!(ACTOR_DECISIONS <= 9300);
const _: () =
    assert!(PPO_ANNEALED_MAX_GAMES * ACTOR_DECISIONS * size_of::<Decision>() < 64 * 1024 * 1024);

impl Decision {
    pub(in crate::ppo_arena) fn capture(
        environment: &TrainingEnvironment,
        choice: &PpoPolicyChoice,
        index: usize,
        retained: bool,
    ) -> Self {
        assert!(index < ACTOR_DECISIONS);
        assert!(choice.value().is_finite());
        let seat = &environment.seats[environment.policy_seat];
        Self {
            tick: seat.tracker.current().unwrap().tick,
            action: choice.action(),
            retained,
            coverage: raze_coverage(seat, choice.action()),
            reward: 0.0,
            ticks: 0,
            terminal: false,
        }
    }
}

fn raze_coverage(seat: &super::super::ArenaSeatPolicy, action: StructuredAction) -> Option<u8> {
    use bota_proto::{Fixed, StatusFlags, UnitKind};
    let StructuredAction::Cast {
        unit: ControlledUnit::Hero,
        slot,
        ..
    } = action
    else {
        return None;
    };
    let hero = seat.tracker.own_hero().unwrap();
    let ability = hero.abilities[usize::from(slot.0)].id.0;
    if !(13..=15).contains(&ability) {
        return None;
    }
    let reach = [200, 450, 700][usize::from(ability - 13)];
    let center = bota_server::game::point_along(
        hero.pos,
        hero.pos + bota_server::game::heading_of(hero.facing),
        Fixed::from_int(reach),
    );
    let mut coverage = 0;
    for unit in &seat.tracker.current().unwrap().units {
        if unit.team == hero.team
            || unit.hp <= 0
            || unit.magic_resist >= Fixed::ONE
            || unit.statuses.bits
                & (StatusFlags::DEAD | StatusFlags::INVULNERABLE | StatusFlags::MAGIC_IMMUNE)
                != 0
            || !unit.pos.within(center, Fixed::from_int(250))
        {
            continue;
        }
        coverage |= match unit.kind {
            UnitKind::Hero => 1,
            UnitKind::CreepMelee
            | UnitKind::CreepFlagbearer
            | UnitKind::CreepRanged
            | UnitKind::CreepSiege
            | UnitKind::CreepNeutral => 2,
            _ => 0,
        };
    }
    Some(coverage)
}

type CensusKey = (
    usize,
    Option<ControlledUnit>,
    Option<usize>,
    bool,
    Option<u8>,
);
fn census_key(row: &Decision) -> CensusKey {
    use StructuredAction::*;
    let slot = match row.action {
        Cast { slot, .. } | Learn { slot } => Some(usize::from(slot.0)),
        Use { slot, .. }
        | Sell { slot, .. }
        | Swap { from: slot, .. }
        | PutPoint { source: slot, .. }
        | PutUnit { source: slot, .. } => Some(usize::from(slot.0)),
        Buy { item, .. } => Some(item.0),
        _ => None,
    };
    (
        row.action.kind().index(),
        row.action.controlled_unit(),
        slot,
        row.tick < crate::MAP2_PREGAME_TICKS,
        row.coverage,
    )
}

#[derive(Default)]
struct Census {
    all: usize,
    retained: usize,
    gae_signs: [usize; 3],
    mc_signs: [usize; 3],
    disagreements: usize,
    gae_minus_mc: f64,
    ticks_to_terminal: u64,
}

fn validate_tape(tape: &[Decision]) {
    assert!(!tape.is_empty());
    assert!(tape.len() <= ACTOR_DECISIONS);
    let mut tick = tape[0].tick;
    for (index, row) in tape.iter().enumerate() {
        assert_eq!(row.tick, tick);
        assert!((1..=MAP2_DECISION_INTERVAL_TICKS).contains(&row.ticks));
        assert_eq!(row.terminal, index + 1 == tape.len());
        assert!(row.reward.is_finite());
        tick += row.ticks;
    }
    assert!(tick <= crate::MAP2_TICK_CAP);
}

fn analyze(batch: &PpoBatch, tapes: &[Vec<Decision>], lambda: f32) -> BTreeMap<CensusKey, Census> {
    assert!((1..=PPO_ANNEALED_MAX_GAMES).contains(&tapes.len()));
    assert_eq!(lambda, 0.98);
    let mut remaining: Vec<_> = tapes
        .iter()
        .map(|tape| (tape.len(), tape.iter().filter(|row| row.retained).count()))
        .collect();
    let mut census = BTreeMap::<CensusKey, Census>::new();
    for tape in tapes {
        validate_tape(tape);
        for row in tape {
            census.entry(census_key(row)).or_default().all += 1;
        }
    }
    let mut returns = [0.0f64; PPO_ANNEALED_MAX_GAMES];
    let mut next_advantage = [0.0f32; PPO_ANNEALED_MAX_GAMES];
    for index in (0..batch.len()).rev() {
        let sample = batch.sample(index).unwrap();
        let transition = &sample.transition;
        let stream = transition.stream;
        let tape = &tapes[stream];
        let (end, retained) = remaining[stream];
        let start = tape[..end].iter().rposition(|row| row.retained).unwrap();
        assert_eq!(transition.decision as usize, retained - 1);
        remaining[stream] = (start, retained - 1);
        let row = &tape[start];
        for row in tape[start..end].iter().rev() {
            returns[stream] += row.reward;
        }
        assert_eq!(transition.action, row.action);
        assert_eq!(
            transition.ticks,
            tape[start..end].iter().map(|row| row.ticks).sum::<u32>()
        );
        assert_eq!(
            transition.reward,
            tape[start..end].iter().map(|row| row.reward).sum::<f64>() as f32
        );
        assert_eq!(transition.terminal, end == tape.len());
        if transition.terminal {
            assert_eq!(transition.next_value, 0.0);
        }
        assert!((f64::from(sample.return_value()) - returns[stream]).abs() < 2e-5);
        let delta = transition.reward + transition.next_value - transition.old_value;
        let gae = retained_advantage(delta, next_advantage[stream], lambda, transition.terminal);
        next_advantage[stream] = gae;
        let mc = returns[stream] - f64::from(transition.old_value);
        let stats = census.get_mut(&census_key(row)).unwrap();
        stats.retained += 1;
        stats.gae_signs[sign(f64::from(gae))] += 1;
        stats.mc_signs[sign(mc)] += 1;
        stats.disagreements += usize::from(sign(f64::from(gae)) != sign(mc));
        stats.gae_minus_mc += f64::from(gae) - mc;
        let last = tape.last().unwrap();
        stats.ticks_to_terminal += u64::from(last.tick + last.ticks - row.tick);
    }
    for (tape, (end, _)) in tapes.iter().zip(remaining) {
        assert!(tape[..end].iter().all(|row| !row.retained));
    }
    census
}

fn sign(value: f64) -> usize {
    match value.partial_cmp(&0.0).expect("finite advantage") {
        std::cmp::Ordering::Less => 0,
        std::cmp::Ordering::Equal => 1,
        std::cmp::Ordering::Greater => 2,
    }
}

fn log_census(census: &BTreeMap<CensusKey, Census>) {
    for (&(kind, body, slot, pregame, coverage), stats) in census {
        eprintln!(
            "learning-census kind={:?} body={body:?} slot={slot:?} pregame={pregame} snapshot_coverage_bits={coverage:?} all={} retained={} gae_signs_neg_zero_pos={:?} mc_signs_neg_zero_pos={:?} sign_disagreements={} mean_gae_minus_mc={:?} mean_ticks_to_terminal={:?}",
            ActionKind::from_index(kind).unwrap(),
            stats.all,
            stats.retained,
            stats.gae_signs,
            stats.mc_signs,
            stats.disagreements,
            (stats.retained > 0).then(|| stats.gae_minus_mc / stats.retained as f64),
            (stats.retained > 0).then(|| stats.ticks_to_terminal as f64 / stats.retained as f64)
        );
    }
}

fn retained_advantage(delta: f32, next: f32, lambda: f32, terminal: bool) -> f32 {
    assert!((0.0..=1.0).contains(&lambda));
    let continuation = if terminal { 0.0 } else { 1.0 };
    let advantage = delta + lambda * next * continuation;
    assert!(advantage.is_finite());
    advantage
}

#[test]
fn raw_credit_preserves_signs_and_resets_at_terminal() {
    assert_eq!([-1.0, -0.0, 0.0, 1.0].map(sign), [0, 1, 1, 2]);
    let terminal = retained_advantage(11.0 - 1.0, 999.0, 0.98, true);
    assert_eq!(terminal, 10.0);
    assert!((retained_advantage(52.0 + 1.0 - 2.0, terminal, 0.98, false) - 60.8).abs() < 1e-5);
    assert_eq!(retained_advantage(51.0, terminal, 1.0, false), 61.0);
    assert_eq!(retained_advantage(99.0, -100.0, 0.98, false), 1.0);
    assert_eq!(retained_advantage(99.0, -100.0, 1.0, false), -1.0);
}

#[test]
fn learning_tape_preserves_collector_rng_actions_and_partial_terminal_returns() {
    let model = PolicyModel::fresh(99103).unwrap();
    let before = model.export_parameters().unwrap();
    let config = episode::parity_settings_for_test().ppo;
    let results = [false, true].map(|enabled| {
        collect(
            &model,
            config,
            (0..2).map(short_environment).collect(),
            actor_stream_rngs(&mut PpoRng::new(81), 2).unwrap(),
            enabled,
        )
    });
    assert_eq!(results[0].random, results[1].random);
    assert_eq!(results[0].report, results[1].report);
    assert_eq!(results[0].traces, results[1].traces);
    episode::assert_batch_parity_for_test(&results[0].batch, &results[1].batch);
    assert_eq!(results[1].report.terminal_draws, 2);
    assert_eq!(results[1].report.elapsed_ticks, 70);
    for tape in &results[1].tapes {
        assert_eq!(tape.len(), 12);
        let last = tape.last().unwrap();
        assert_eq!(last.ticks, 2);
        assert_eq!(last.tick + last.ticks, crate::MAP2_TICK_CAP);
    }
    let census = analyze(&results[1].batch, &results[1].tapes, config.gae_lambda);
    assert_eq!(census.values().map(|stats| stats.all).sum::<usize>(), 24);
    assert_eq!(
        census.values().map(|stats| stats.retained).sum::<usize>(),
        results[1].batch.len()
    );
    crate::tests::support::assert_bits(&model.export_parameters().unwrap(), &before);
}

fn short_environment(side: usize) -> TrainingEnvironment {
    let mut environment =
        build_environment(982001, 982007, MapId(2), side, 0, OpponentSpec::Weak).unwrap();
    let (_, mut start) = crate::Arena::new(crate::ArenaConfig {
        seats: 2,
        map: MapId(2),
        seed: 982001,
    })
    .unwrap();
    let configured = environment
        .arena
        .configure_for_test(|world| world.tick = crate::MAP2_TICK_CAP - 35);
    for (messages, current) in start.messages.iter_mut().zip(configured.messages) {
        messages.truncate(1);
        messages.extend(current);
    }
    environment.seats = setup_seats(start).unwrap();
    environment
}
