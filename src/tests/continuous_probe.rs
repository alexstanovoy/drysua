use super::*;

#[test]
fn continuous_segments_preserve_live_game_rng_and_action_history() {
    let model = PolicyModel::fresh(10083000).expect("model");
    let settings = PpoSmokeConfig {
        environments: 2,
        map: MapId(0),
        seed: 10083001,
        ..PpoSmokeConfig::default()
    };
    let mut source = build_environments(settings).expect("source");
    let mut target = build_environments(settings).expect("target");
    let mut source_rng = actor_stream_rngs(&mut PpoRng::new(10083002), 2).expect("source RNG");
    let mut target_rng = source_rng.clone();
    let config = PpoConfig {
        environments: 2,
        rollout_decisions: 8,
        minibatch: 8,
        ..PpoConfig::default()
    };
    let mut whole =
        PpoRollout::new(16, model.policy_identity().expect("identity")).expect("rollout");
    collect_segment(
        &model,
        &mut source_rng,
        &mut source,
        config,
        8,
        &mut whole,
        &mut PpoSmokeReport::default(),
    )
    .expect("whole");
    for _ in 0..2 {
        let mut half =
            PpoRollout::new(8, model.policy_identity().expect("identity")).expect("half");
        collect_segment(
            &model,
            &mut target_rng,
            &mut target,
            config,
            4,
            &mut half,
            &mut PpoSmokeReport::default(),
        )
        .expect("half");
        assert_eq!(half.len(), 8);
    }
    assert_eq!(source_rng, target_rng);
    for index in 0..2 {
        assert_eq!(target[index].arena.tick(), 25);
        assert_eq!(target[index].decision, 8);
        assert_eq!(
            prepare_policy_sample(&mut source[index]).expect("source").0,
            prepare_policy_sample(&mut target[index]).expect("target").0
        );
        assert_eq!(
            source[index].seats[0].sequence,
            target[index].seats[0].sequence
        );
        assert_eq!(
            source[index].seats[1].sequence,
            target[index].seats[1].sequence
        );
    }
}

fn collect_segment(
    model: &PolicyModel,
    random: &mut [PpoRng],
    arenas: &mut [TrainingEnvironment],
    config: PpoConfig,
    decisions: usize,
    rollout: &mut PpoRollout,
    report: &mut PpoSmokeReport,
) -> Result<(), PpoError> {
    assert!((1..=512).contains(&decisions));
    assert_eq!(random.len(), arenas.len());
    for _ in 0..decisions {
        let pending = collect_round(model, random, arenas, config)?;
        let bootstrap = bootstrap_values(model, &pending)?;
        commit_round(arenas, pending, bootstrap, rollout, report)?;
    }
    Ok(())
}

#[test]
#[ignore = "Bounded hypothesis probe of existing PPO engine with persistent games; no release gate"]
fn probe_continuous_ppo_learning() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let output =
        std::path::PathBuf::from(std::env::var("DRYSUA_CONTINUOUS_OUTPUT").expect("output"));
    assert!(
        output.canonicalize().expect("existing output").starts_with(
            root.join("artifacts/temp")
                .canonicalize()
                .expect("artifact root")
        )
    );
    assert!(output.read_dir().expect("output entries").next().is_none());
    let seed = std::env::var("DRYSUA_CONTINUOUS_SEED")
        .unwrap_or_else(|_| "10083010".into())
        .parse::<u64>()
        .expect("seed");
    assert!((10083010..=10083019).contains(&seed));
    let weights = root.join("artifacts/temp/input-facts-m12-u10-init");
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    let device = PolicyDevice::Cuda { ordinal: 0 };
    #[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
    let device = PolicyDevice::Cpu;
    let model = PolicyModel::fresh_on(seed, device).expect("model");
    TrainingArtifact::load_runtime_weights(&model, &weights).expect("initializer");
    let mut probe = ContinuousProbe::new(model, seed, continuous_config());
    let started = Instant::now();
    eprintln!(
        "continuous_probe config={:?} seed={seed} reward=existing_bounded_shaping terminal=existing_plus_minus_one no_time_cost=true updates_limit=64 policy_lag=0 qualification=false",
        probe.config
    );
    for update in 1..=64 {
        let learning = probe.step();
        probe.print_update(update, started.elapsed(), &learning);
        if update % 16 == 0 {
            probe.save(&output, seed, update, &learning);
        }
        assert!(probe.arenas.iter().all(|arena| arena.arena.tick() < 108900));
        assert!(
            started.elapsed() < Duration::from_secs(1800),
            "continuous hypothesis budget exhausted"
        );
    }
    assert_eq!(probe.trainer.updates(), 64);
    assert!(probe.report.elapsed_ticks <= 64 * 4096 * 3);
    assert!(probe.report.elapsed_ticks >= 64 * 4096);
    eprintln!(
        "continuous_finished report={:?} seconds={:.3}",
        probe.report,
        started.elapsed().as_secs_f64()
    );
}

fn continuous_config() -> PpoConfig {
    PpoConfig {
        environments: 8,
        rollout_decisions: 512,
        epochs: 1,
        minibatch: 512,
        learning_rate: 3e-5,
        entropy_coefficient: 0.001,
        gamma_tick: 1.0,
        gae_lambda: 0.95,
        ..PpoConfig::default()
    }
}

struct ContinuousProbe {
    model: PolicyModel,
    config: PpoConfig,
    arenas: Vec<TrainingEnvironment>,
    random: Vec<PpoRng>,
    trainer: PpoTrainer,
    report: PpoSmokeReport,
}

impl ContinuousProbe {
    fn new(model: PolicyModel, seed: u64, config: PpoConfig) -> Self {
        assert!(config.environments <= 8);
        assert!(config.environments.is_multiple_of(2));
        let arenas = (0..config.environments as u64)
            .map(|stream| {
                build_environment(
                    seed + stream / 2,
                    seed ^ 0x6f70706f6e656e74,
                    MapId(0),
                    stream as usize % 2,
                    0,
                    OpponentSpec::Teacher,
                )
                .expect("paired persistent game")
            })
            .collect();
        let random = actor_stream_rngs(&mut PpoRng::new(seed ^ 0x6163746f72), config.environments)
            .expect("actor RNGs");
        let trainer =
            PpoTrainer::new(&model, config, seed ^ 0x6c6561726e).expect("fresh optimizer");
        Self {
            model,
            config,
            arenas,
            random,
            trainer,
            report: PpoSmokeReport::default(),
        }
    }

    fn step(&mut self) -> PpoUpdateReport {
        let capacity = self.config.environments * self.config.rollout_decisions;
        let mut rollout =
            PpoRollout::new(capacity, self.model.policy_identity().expect("identity"))
                .expect("rollout");
        collect_segment(
            &self.model,
            &mut self.random,
            &mut self.arenas,
            self.config,
            self.config.rollout_decisions,
            &mut rollout,
            &mut self.report,
        )
        .expect("segment");
        assert_eq!(rollout.len(), capacity);
        let batch = rollout.finish(self.config).expect("batch");
        let learning = self
            .trainer
            .train_update(&self.model, &batch)
            .expect("update");
        record_update(&mut self.report, capacity, learning).expect("learning counters");
        assert_eq!(self.report.optimizer_step, self.trainer.optimizer_step());
        learning
    }

    fn print_update(&self, update: u32, elapsed: Duration, learning: &PpoUpdateReport) {
        assert_eq!(self.trainer.updates(), u64::from(update));
        assert!(update > 0);
        eprintln!(
            "continuous_update update={update} seconds={:.3} wins={} losses={} ticks={} arena_ticks={:?} learning={learning:?}",
            elapsed.as_secs_f64(),
            self.report.terminal_wins,
            self.report.terminal_losses,
            self.report.elapsed_ticks,
            self.arenas
                .iter()
                .map(|arena| arena.arena.tick())
                .collect::<Vec<_>>()
        );
    }

    fn save(&self, output: &Path, seed: u64, update: u32, learning: &PpoUpdateReport) {
        assert_eq!(self.trainer.updates(), u64::from(update));
        assert!(update.is_multiple_of(16));
        let checkpoint = output.join(format!("update-{update:03}"));
        std::fs::create_dir(&checkpoint).expect("checkpoint directory");
        TrainingArtifact::save_runtime_weights(&self.model, &checkpoint).expect("weights");
        std::fs::write(checkpoint.join("probe.txt"), format!("experimental_existing_ppo_engine=true\nresume_supported=false\nconfig={:?}\nseed={seed}\nupdate={update}\n{:?}\n{learning:?}\n", self.config, self.report)).expect("manifest");
    }
}

#[test]
fn continuous_report_records_optimizer_updates_and_samples() {
    let model = PolicyModel::fresh(10083005).expect("model");
    let config = PpoConfig {
        environments: 2,
        rollout_decisions: 2,
        minibatch: 4,
        epochs: 1,
        ..PpoConfig::default()
    };
    let mut probe = ContinuousProbe::new(model, 10083005, config);
    probe.step();
    assert_eq!(probe.report.updates, 1);
    assert_eq!(probe.report.transitions, 4);
    assert_eq!(probe.report.optimizer_step, 1);
}
