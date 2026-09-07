#![allow(
    clippy::float_arithmetic,
    reason = "Discounted n-step rewards use floating-point arithmetic"
)]

use super::*;

pub(super) const TICK_CAP: u32 = 108_900;
pub(super) const ACTOR_DECISIONS: usize = 36_300;
const RETENTION_STRIDE: usize = 8;
const RETAINED_PER_EPISODE: usize = ACTOR_DECISIONS.div_ceil(RETENTION_STRIDE);
const MAX_EPISODE_ENVIRONMENTS: usize = 6;
const RETAINED_BYTES_PER_ENVIRONMENT: usize = 3
    * RETAINED_PER_EPISODE
    * (std::mem::size_of::<FeatureFrame>() + std::mem::size_of::<crate::BehavioralTarget>());
const _: () =
    assert!(MAX_EPISODE_ENVIRONMENTS * RETAINED_BYTES_PER_ENVIRONMENT < 6 * 1024 * 1024 * 1024);
const _: () = assert!(MAX_EPISODE_ENVIRONMENTS * RETAINED_PER_EPISODE <= crate::PPO_MAX_SAMPLES);

pub(crate) fn validate(settings: &TrainingJobConfig) -> Result<(), PpoError> {
    if settings.terminal_only && !settings.complete_episodes {
        return Err(PpoError::InvalidConfig(
            "terminal-only requires complete episodes",
        ));
    }
    if !settings.complete_episodes {
        return Ok(());
    }
    if settings.map != MapId(0)
        || !matches!(settings.ppo.environments, 2 | 4 | 6)
        || settings.ppo.decision_interval_ticks != 3
    {
        return Err(PpoError::InvalidConfig(
            "complete episodes require Map0, two/four/six environments, and three-tick actions",
        ));
    }
    if settings.ppo.rollout_decisions < RETAINED_PER_EPISODE {
        return Err(PpoError::InvalidConfig(
            "complete episode retained capacity",
        ));
    }
    assert!(settings.ppo.environments * RETAINED_BYTES_PER_ENVIRONMENT < 6 * 1024 * 1024 * 1024);
    settings.ppo.validate()?;
    Ok(())
}

pub(super) fn environments(
    settings: &TrainingJobConfig,
    update: u64,
) -> Result<Vec<TrainingEnvironment>, PpoError> {
    validate(settings)?;
    assert!(settings.complete_episodes);
    let first_pair = update
        .checked_mul((settings.ppo.environments / 2) as u64)
        .ok_or(PpoError::CounterOverflow)?;
    (0..settings.ppo.environments)
        .map(|stream| {
            let pair = first_pair
                .checked_add((stream / 2) as u64)
                .ok_or(PpoError::CounterOverflow)?;
            build_environment(
                derive_training_seed(settings.seed, pair, 0x6172_656e_615f_7365),
                derive_training_seed(settings.seed, pair, 0x6f70_706f_6e65_6e74),
                MapId(0),
                stream % 2,
                0,
                OpponentSpec::Teacher,
            )
        })
        .collect()
}

pub(super) fn collect(
    model: &PolicyModel,
    sampling: &mut PpoRng,
    environments: &mut [TrainingEnvironment],
    config: PpoConfig,
    rollout: &mut PpoRollout,
    report: &mut PpoSmokeReport,
    terminal_only: bool,
) -> Result<(), PpoError> {
    assert!(matches!(environments.len(), 2 | 4 | 6));
    assert_eq!(config.decision_interval_ticks, 3);
    let mut random = actor_stream_rngs(sampling, environments.len())?;
    let mut streams: Vec<_> = (0..environments.len())
        .map(|_| EpisodeStream {
            terminal_only,
            ..EpisodeStream::default()
        })
        .collect();
    let retained_bytes_bound = environments.len() * RETAINED_BYTES_PER_ENVIRONMENT;
    eprintln!(
        "collection: complete-episodes opponent=Teacher actor_ticks=3 retention_stride=8 tick_cap={TICK_CAP} environments={} terminal_only={terminal_only} retained_bytes_bound={retained_bytes_bound}",
        environments.len()
    );
    for _ in 0..ACTOR_DECISIONS {
        let active: Vec<_> = (0..streams.len())
            .filter(|&stream| !streams[stream].done)
            .collect();
        if active.is_empty() {
            break;
        }
        let (choices, spaces) = sample_active(model, &mut random, environments, &active)?;
        for ((stream, choice), space) in active.into_iter().zip(choices).zip(spaces) {
            advance_stream(
                model,
                &mut environments[stream],
                &mut streams[stream],
                (stream, choice, space),
                config,
                rollout,
                report,
            )?;
        }
    }
    assert!(streams.iter().all(|stream| stream.done));
    assert!(streams.iter().all(|stream| stream.choice.is_none()));
    report.rejected_orders = environment_rejections(environments)?;
    if report.terminal_wins + report.terminal_losses == 0 {
        return Err(PpoError::InvalidTransition(
            "complete episode batch has no authoritative terminal outcomes",
        ));
    }
    Ok(())
}

fn sample_active(
    model: &PolicyModel,
    random: &mut [PpoRng],
    environments: &mut [TrainingEnvironment],
    active: &[usize],
) -> Result<(Vec<PpoPolicyChoice>, Vec<ActionSpace>), PpoError> {
    assert!(!active.is_empty());
    assert!(active.len() <= MAX_EPISODE_ENVIRONMENTS);
    if random.len() != environments.len()
        || active.iter().any(|index| *index >= environments.len())
        || active.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(PpoError::InvalidConfig("episode active streams"));
    }
    let mut frames = Vec::with_capacity(active.len());
    let mut spaces = Vec::with_capacity(active.len());
    let mut selected_random = Vec::with_capacity(active.len());
    for &stream in active {
        let (frame, space) = prepare_policy_sample(&mut environments[stream])?;
        frames.push(frame);
        spaces.push(space);
        selected_random.push(random[stream].clone());
    }
    let choices = model
        .sample_batch(&frames, &spaces, &mut selected_random)
        .map_err(model_error)?;
    for (&stream, state) in active.iter().zip(selected_random) {
        random[stream] = state;
    }
    Ok((choices, spaces))
}

#[derive(Default)]
struct EpisodeStream {
    terminal_only: bool,
    elapsed_ticks: u32,
    raw_return: f64,
    discounted_return: f64,
    shaping_return: f64,
    terminal_reward: f32,
    actions: [u32; ActionKind::COUNT],
    choice: Option<PpoPolicyChoice>,
    interval: DiscountedInterval,
    decisions: usize,
    retained: u32,
    done: bool,
}

#[derive(Default)]
struct DiscountedInterval {
    reward: f64,
    ticks: u32,
    steps: usize,
}

impl DiscountedInterval {
    fn append(&mut self, reward: f32, ticks: u32, gamma: f32) -> Result<(), PpoError> {
        if !reward.is_finite() {
            return Err(PpoError::NonFinite("episode reward"));
        }
        let _ = tick_discount(gamma, ticks)?;
        assert!(self.steps < RETENTION_STRIDE);
        let discount = if self.ticks == 0 {
            1.0
        } else {
            tick_discount(gamma, self.ticks)?
        };
        self.reward += f64::from(discount) * f64::from(reward);
        self.ticks = self
            .ticks
            .checked_add(ticks)
            .ok_or(PpoError::CounterOverflow)?;
        self.steps += 1;
        assert!(self.reward.is_finite());
        Ok(())
    }

    fn finish(self, stream: usize, decision: u32, terminal: bool, next_value: f32) -> PpoOutcome {
        assert!(self.steps > 0);
        assert!(self.steps <= RETENTION_STRIDE);
        PpoOutcome {
            stream,
            decision,
            ticks: self.ticks,
            reward: self.reward as f32,
            terminal,
            next_value: if terminal { 0.0 } else { next_value },
        }
    }
}

fn advance_stream(
    model: &PolicyModel,
    environment: &mut TrainingEnvironment,
    state: &mut EpisodeStream,
    sample: (usize, PpoPolicyChoice, ActionSpace),
    config: PpoConfig,
    rollout: &mut PpoRollout,
    report: &mut PpoSmokeReport,
) -> Result<(), PpoError> {
    let (stream, choice, space) = sample;
    assert!(!state.done);
    assert!(state.decisions < ACTOR_DECISIONS);
    if state.choice.is_none() {
        state.choice = Some(choice.clone());
    }
    let tick = environment.seats[environment.policy_seat]
        .tracker
        .current()
        .ok_or(PpoError::InvalidTransition("episode snapshot"))?
        .tick;
    assert!(tick < TICK_CAP);
    let requests = requests_for_decision_in_space(environment, &choice, &space)?;
    let advanced = advance_interval(environment, requests, 3.min(TICK_CAP - tick))?;
    reject_production_rejection(environment, "complete episode rollout")?;
    let outcome = terminal_outcome(environment, advanced.winner);
    let reward = observe_reward(
        environment,
        state,
        outcome,
        advanced.ticks,
        config.gamma_tick,
    )?;
    state.actions[choice.action().kind().index()] += 1;
    state
        .interval
        .append(reward, advanced.ticks, config.gamma_tick)?;
    state.decisions += 1;
    report.elapsed_ticks = report
        .elapsed_ticks
        .checked_add(u64::from(advanced.ticks))
        .ok_or(PpoError::CounterOverflow)?;
    state.done = outcome.is_some() || tick + advanced.ticks >= TICK_CAP;
    if state.interval.steps == RETENTION_STRIDE || state.done {
        flush(
            model,
            environment,
            state,
            stream,
            outcome.is_some(),
            rollout,
        )?;
    }
    if state.done {
        record_episode(stream, tick + advanced.ticks, state, outcome, report)?;
    }
    Ok(())
}

fn observe_reward(
    environment: &mut TrainingEnvironment,
    state: &mut EpisodeStream,
    outcome: Option<PpoTerminalOutcome>,
    ticks: u32,
    gamma: f32,
) -> Result<f32, PpoError> {
    assert!(ticks > 0);
    assert!(state.elapsed_ticks < TICK_CAP);
    let reward = if state.terminal_only {
        RewardTracker::terminal_only(outcome)?
    } else {
        let summary = environment.seats[environment.policy_seat]
            .tracker
            .latest_summary()
            .ok_or(PpoError::InvalidTransition("episode next summary"))?;
        environment
            .reward
            .observe(summary, tick_discount(gamma, ticks)?, outcome)?
    };
    state.raw_return += f64::from(reward.total);
    state.discounted_return +=
        f64::from(gamma).powi(state.elapsed_ticks as i32) * f64::from(reward.total);
    state.shaping_return += f64::from(reward.total - reward.terminal);
    state.terminal_reward = reward.terminal;
    state.elapsed_ticks += ticks;
    Ok(reward.total)
}

fn flush(
    model: &PolicyModel,
    environment: &mut TrainingEnvironment,
    state: &mut EpisodeStream,
    stream: usize,
    terminal: bool,
    rollout: &mut PpoRollout,
) -> Result<(), PpoError> {
    assert!(state.retained < RETAINED_PER_EPISODE as u32);
    let next_value = if terminal {
        0.0
    } else {
        model
            .evaluate_batch(&[encode_next_frame(environment)?])
            .map_err(model_error)?[0]
            .value
    };
    let choice = state
        .choice
        .take()
        .ok_or(PpoError::InvalidTransition("episode retained action"))?;
    assert_eq!(choice.policy(), rollout.policy());
    let outcome =
        std::mem::take(&mut state.interval).finish(stream, state.retained, terminal, next_value);
    rollout.push(choice.finish(outcome)?)?;
    state.retained += 1;
    Ok(())
}

fn record_episode(
    stream: usize,
    tick: u32,
    state: &EpisodeStream,
    outcome: Option<PpoTerminalOutcome>,
    report: &mut PpoSmokeReport,
) -> Result<(), PpoError> {
    assert!(state.done);
    assert!(tick <= TICK_CAP);
    let (label, counter) = match outcome {
        Some(PpoTerminalOutcome::Win) => ("Win", &mut report.terminal_wins),
        Some(PpoTerminalOutcome::Loss) => ("Loss", &mut report.terminal_losses),
        Some(PpoTerminalOutcome::Draw) => {
            return Err(PpoError::InvalidTransition("Map0 episode draw"));
        }
        None => ("Timeout", &mut report.episode_timeouts),
    };
    *counter = counter.checked_add(1).ok_or(PpoError::CounterOverflow)?;
    eprintln!(
        "episode: stream={stream} opponent=Teacher tick={tick} outcome={label} actor_decisions={} retained={} terminal_sample={} terminal_only={} raw_return={:.9} discounted_return={:.9} terminal_reward={} shaping_return={:.9} actions={:?} noncontinue={}",
        state.decisions,
        state.retained,
        outcome.is_some(),
        state.terminal_only,
        state.raw_return,
        state.discounted_return,
        state.terminal_reward,
        state.shaping_return,
        state.actions,
        state.decisions - state.actions[ActionKind::Continue.index()] as usize
    );
    Ok(())
}

#[cfg(test)]
pub(crate) fn assert_discounted_intervals_for_test() {
    let mut interval = DiscountedInterval::default();
    interval.append(2.0, 2, 0.5).expect("first interval");
    interval.append(4.0, 1, 0.5).expect("second interval");
    let outcome = interval.finish(0, 0, true, 99.0);
    assert_eq!(outcome.reward, 3.0);
    assert_eq!(outcome.ticks, 3);
    assert!(outcome.terminal);
    assert_eq!(outcome.next_value, 0.0);
    let mut partial = DiscountedInterval::default();
    partial.append(-1.0, 1, 0.9).expect("last partial");
    let outcome = partial.finish(0, 1, false, 0.25);
    assert!(!outcome.terminal);
    assert_eq!(outcome.next_value, 0.25);
    assert_eq!(outcome.ticks, 1);
    assert_eq!(outcome.reward, -1.0);
    let mut invalid = DiscountedInterval::default();
    assert_eq!(
        invalid
            .append(f32::NAN, 3, 0.9)
            .expect_err("NaN")
            .to_string(),
        "PPO episode reward is non-finite"
    );
}

#[cfg(test)]
pub(crate) fn assert_reset_loses_terminal_credit_for_test() {
    let delayed_terminal_tick = 20_000;
    let maximum_window_end = 1 + training_warmup_decisions(7) as u32 * 3 + 6 + 2048 * 3;
    assert!(maximum_window_end < delayed_terminal_tick);
    let mut reset_terminals = 0;
    let mut continuous_terminals = 0;
    let mut continuous_tick = 1;
    for _ in 0..4 {
        let reset_tick = 1 + 2048 * 3;
        reset_terminals += usize::from(reset_tick >= delayed_terminal_tick);
        let previous = continuous_tick;
        continuous_tick += 2048 * 3;
        continuous_terminals += usize::from(
            previous < delayed_terminal_tick && continuous_tick >= delayed_terminal_tick,
        );
    }
    assert_eq!(reset_terminals, 0);
    assert_eq!(continuous_terminals, 1);
    assert!(TICK_CAP > delayed_terminal_tick);
    let settings = crate::cli::training_settings_for_test(&["--map", "0", "--environments", "2"])
        .expect("windows");
    let model = PolicyModel::fresh(9911000).expect("model");
    let mut initial =
        build_training_environments(&settings, 0, settings.ppo, &model).expect("first windows");
    let start_tick = initial[0].seats[0].tracker.current().expect("start").tick;
    advance_interval(&mut initial[0], vec![None, None], 3).expect("progress");
    let progressed_tick = initial[0].seats[0]
        .tracker
        .current()
        .expect("progressed")
        .tick;
    let rebuilt =
        build_training_environments(&settings, 8, settings.ppo, &model).expect("new windows");
    assert_eq!(
        rebuilt[0].seats[0].tracker.current().expect("rebuilt").tick,
        start_tick
    );
    assert!(progressed_tick > start_tick);
}

#[cfg(test)]
pub(crate) fn assert_rng_and_sample_provenance_for_test(model: &PolicyModel) {
    let settings = crate::cli::training_settings_for_test(&[
        "--complete-episodes",
        "--map",
        "0",
        "--environments",
        "2",
        "--rollout",
        "8192",
    ])
    .expect("settings");
    let master = PpoRng::new(9911001);
    let mut first = master.clone();
    let (state, draws) = master.checkpoint();
    let mut second = PpoRng::from_checkpoint(state, draws).expect("restored master");
    let source = initial_interval_for_test(model, &settings, &mut first);
    let target = initial_interval_for_test(model, &settings, &mut second);
    assert_eq!(source, target);
    assert_eq!(first, second);
}

#[cfg(test)]
fn initial_interval_for_test(
    model: &PolicyModel,
    settings: &TrainingJobConfig,
    master: &mut PpoRng,
) -> Vec<(crate::StructuredAction, f32, u32)> {
    let mut arenas = environments(settings, 0).expect("arenas");
    let mut random = actor_stream_rngs(master, 2).expect("random streams");
    let mut streams = [EpisodeStream::default(), EpisodeStream::default()];
    let mut rollout =
        PpoRollout::new(2, model.policy_identity().expect("identity")).expect("rollout");
    let mut report = PpoSmokeReport::default();
    let mut expected = Vec::new();
    for step in 0..RETENTION_STRIDE {
        let (choices, spaces) =
            sample_active(model, &mut random, &mut arenas, &[0, 1]).expect("choices");
        if step == 0 {
            expected = choices
                .iter()
                .map(|choice| (choice.action(), choice.log_probability()))
                .collect();
        }
        for (stream, (choice, space)) in choices.into_iter().zip(spaces).enumerate() {
            advance_stream(
                model,
                &mut arenas[stream],
                &mut streams[stream],
                (stream, choice, space),
                settings.ppo,
                &mut rollout,
                &mut report,
            )
            .expect("act every tick interval");
        }
    }
    assert_eq!(rollout.len(), 2);
    let batch = rollout.finish(settings.ppo).expect("batch");
    (0..2)
        .map(|index| {
            let sample = batch.sample(index).expect("sample").transition;
            assert_eq!((sample.action, sample.old_log_probability), expected[index]);
            assert_eq!(sample.ticks, 24);
            assert!(!sample.terminal);
            (sample.action, sample.old_log_probability, sample.ticks)
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn assert_checkpoint_scope_for_test(mut settings: TrainingJobConfig, directory: &Path) {
    let config = settings.ppo;
    let run = training_checkpoint_run(&settings, PolicyDevice::Cpu, config).expect("run");
    assert!(run.command_line.ends_with(" --complete-episodes"));
    let session = TrainingSession::initialize(
        &settings,
        PolicyDevice::Cpu,
        directory,
        false,
        None,
        config,
        run.clone(),
    )
    .expect("session");
    session
        .save(directory, session.checkpoint_report(None))
        .expect("checkpoint");
    let restored_model = PolicyModel::fresh(9911002).expect("unowned model");
    let restored = restore_training_session(
        &restored_model,
        directory,
        &run,
        config,
        ResumeProvenance::Strict,
    )
    .expect("restore");
    assert_eq!(restored.0.config(), config);
    assert_eq!(restored.1, session.sampling);
    settings.terminal_only = true;
    let terminal =
        training_checkpoint_run(&settings, PolicyDevice::Cpu, config).expect("terminal run");
    assert_eq!(
        TrainingArtifact::load_compatible(directory, &terminal).expect_err("reward mismatch"),
        crate::CheckpointError::InvalidManifest("compatibility scope")
    );
    settings.terminal_only = false;
    settings.complete_episodes = false;
    let windows =
        training_checkpoint_run(&settings, PolicyDevice::Cpu, config).expect("window run");
    assert_eq!(
        TrainingArtifact::load_compatible(directory, &windows).expect_err("mode mismatch"),
        crate::CheckpointError::InvalidManifest("compatibility scope")
    );
}

#[cfg(test)]
pub(crate) fn assert_full_mc_for_test() {
    let model = PolicyModel::fresh(9921000).expect("model");
    let mut arena =
        build_environment(9921001, 9921002, MapId(0), 0, 0, OpponentSpec::Teacher).expect("arena");
    let choice = sample_policy(&model, &mut PpoRng::new(9921003), &mut arena).expect("choice");
    let config = PpoConfig {
        environments: 2,
        rollout_decisions: RETAINED_PER_EPISODE,
        minibatch: 512,
        gae_lambda: 1.0,
        gamma_tick: 0.9999722,
        ..PpoConfig::default()
    };
    let mut rollout = PpoRollout::new(2 * RETAINED_PER_EPISODE, choice.policy()).expect("rollout");
    for decision in 0..RETAINED_PER_EPISODE {
        let count = (ACTOR_DECISIONS - decision * RETENTION_STRIDE).min(RETENTION_STRIDE);
        let terminal = decision + 1 == RETAINED_PER_EPISODE;
        for stream in 0..2 {
            let mut sample = choice.clone();
            sample.value = if decision.is_multiple_of(2) {
                17.0
            } else {
                -31.0
            };
            let sign = if stream == 0 { 1.0 } else { -1.0 };
            let reward = if terminal {
                sign * tick_discount(config.gamma_tick, (count as u32 - 1) * 3)
                    .expect("within interval")
            } else {
                0.0
            };
            let ticks = count as u32 * 3 - u32::from(terminal);
            rollout
                .push(
                    sample
                        .finish(PpoOutcome {
                            stream,
                            decision: decision as u32,
                            ticks,
                            reward,
                            terminal,
                            next_value: if terminal { 0.0 } else { 99.0 },
                        })
                        .expect("transition"),
                )
                .expect("push");
        }
    }
    let batch = rollout.finish(config).expect("MC batch");
    for index in [0, 1, 2000, 4001, batch.len() - 2, batch.len() - 1] {
        let sample = batch.sample(index).expect("sample");
        let sign = if index.is_multiple_of(2) { 1.0 } else { -1.0 };
        let exponent = 108897 - (index / 2 * 24) as i32;
        let expected = sign * f64::from(config.gamma_tick).powi(exponent);
        assert!((f64::from(sample.return_value()) - expected).abs() < 1.0e-5);
    }
    let mut truncated = PpoRollout::new(1, choice.policy()).expect("truncated rollout");
    truncated
        .push(
            choice
                .finish(PpoOutcome {
                    stream: 0,
                    decision: 0,
                    ticks: 11,
                    next_value: 0.7,
                    reward: 0.0,
                    terminal: false,
                })
                .expect("timeout"),
        )
        .expect("push");
    let sample = truncated
        .finish(config)
        .expect("timeout MC")
        .sample(0)
        .expect("sample");
    assert!(!sample.transition.terminal);
    assert!(
        (f64::from(sample.return_value())
            - f64::from(config.gamma_tick).powi(11) * f64::from(0.7f32))
        .abs()
            < 1e-5
    );
}

#[cfg(test)]
pub(crate) fn assert_ragged_streams_for_test(model: &PolicyModel) {
    for count in [2, 4, 6] {
        let mut settings = crate::cli::training_settings_for_test(&[
            "--complete-episodes",
            "--map",
            "0",
            "--environments",
            "2",
            "--rollout",
            "4538",
        ])
        .expect("settings");
        settings.ppo.environments = count;
        validate(&settings).expect("bounded paired batch");
        if count == 2 {
            for update in [0, 1, 17] {
                let arenas = environments(&settings, update).expect("legacy E2 seed mapping");
                for (side, arena) in arenas.iter().enumerate() {
                    assert_eq!(arena.policy_seat, side);
                    assert_eq!(
                        arena.next_seed,
                        derive_training_seed(settings.seed, update, 0x6172_656e_615f_7365)
                            .wrapping_add(1u64 << 32)
                    );
                }
            }
        }
        let first = ragged_trial_for_test(model, &settings);
        let second = ragged_trial_for_test(model, &settings);
        assert_eq!(first, second);
    }
}

#[cfg(test)]
fn ragged_trial_for_test(model: &PolicyModel, settings: &TrainingJobConfig) -> Vec<(u32, f32)> {
    let count = settings.ppo.environments;
    let mut arenas = environments(settings, 0).expect("arenas");
    let mut random = actor_stream_rngs(&mut PpoRng::new(9921004), count).expect("streams");
    let mut states: Vec<_> = (0..count).map(|_| EpisodeStream::default()).collect();
    let mut rollout =
        PpoRollout::new(count, model.policy_identity().expect("identity")).expect("rollout");
    let mut first_logprobs = vec![0.0; count];
    for step in 0..8 {
        let active: Vec<_> = (0..count).filter(|&stream| !states[stream].done).collect();
        if active.is_empty() {
            break;
        }
        let before = random.clone();
        let (choices, spaces) =
            sample_active(model, &mut random, &mut arenas, &active).expect("active batch");
        for stream in 0..count {
            if states[stream].done {
                assert_eq!(random[stream], before[stream]);
            }
        }
        for ((stream, choice), space) in active.into_iter().zip(choices).zip(spaces) {
            if step == 0 {
                first_logprobs[stream] = choice.log_probability();
            }
            advance_stream(
                model,
                &mut arenas[stream],
                &mut states[stream],
                (stream, choice, space),
                settings.ppo,
                &mut rollout,
                &mut PpoSmokeReport::default(),
            )
            .expect("step");
            if step + 1 == 3 + stream % 3 {
                states[stream].done = true;
                flush(
                    model,
                    &mut arenas[stream],
                    &mut states[stream],
                    stream,
                    true,
                    &mut rollout,
                )
                .expect("scripted terminal flush");
            }
        }
    }
    assert!(
        states
            .iter()
            .all(|state| state.done && state.choice.is_none())
    );
    assert_eq!(rollout.len(), count);
    let batch = rollout.finish(settings.ppo).expect("batch");
    (0..count)
        .map(|index| {
            let sample = batch.sample(index).expect("sample").transition;
            assert!(sample.terminal);
            assert_eq!(sample.next_value, 0.0);
            assert_eq!(sample.old_log_probability, first_logprobs[sample.stream]);
            (sample.ticks, sample.old_log_probability)
        })
        .collect()
}
