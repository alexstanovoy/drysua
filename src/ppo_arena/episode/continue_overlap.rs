use super::*;
use crate::ppo_arena::OpponentRuntime;

enum Job {
    Prepare,
    Advance {
        state: Box<EpisodeStream>,
        choice: Box<PpoPolicyChoice>,
        space: Option<Box<ActionSpace>>,
        tick: u32,
    },
}

struct Reply {
    state: EpisodeStream,
    completed: CompletedAdvance,
    prepared: Option<(FeatureFrame, ActionSpace)>,
    opponent: &'static str,
}

enum Output {
    Prepared(Box<(FeatureFrame, ActionSpace)>),
    Advanced(Box<Reply>),
}

type ValuedReply = (Box<Reply>, Option<f32>);

type Workers<'scope> =
    super::super::parallel::StreamWorkers<'scope, TrainingEnvironment, Job, Output>;

#[allow(clippy::too_many_arguments)]
pub(super) fn collect(
    model: &PolicyModel,
    config: PpoConfig,
    stream_base: usize,
    environments: &mut [TrainingEnvironment],
    streams: &mut [EpisodeStream],
    random: &mut [PpoRng],
    rounds: usize,
    rollout: &mut PpoRollout,
    report: &mut PpoSmokeReport,
    #[cfg(test)] fail_late: bool,
) -> Result<(), PpoError> {
    if environments.is_empty()
        || environments.len() > crate::PPO_ANNEALED_MAX_GAMES
        || rounds == 0
        || rounds > ACTOR_DECISIONS
        || stream_base
            .checked_add(environments.len())
            .is_none_or(|end| end > config.environments)
    {
        return Err(PpoError::InvalidConfig("Continue overlap dimensions"));
    }
    if environments.iter().any(|world| {
        !matches!(
            world.opponent,
            OpponentRuntime::Teacher | OpponentRuntime::Weak
        )
    }) {
        return Err(PpoError::InvalidConfig(
            "Continue overlap requires a scripted opponent",
        ));
    }
    assert_eq!(environments.len(), streams.len());
    assert_eq!(streams.len(), random.len());
    std::thread::scope(|scope| {
        let workers = Workers::spawn(scope, environments, "continue-v1", |_, world, job| {
            operate(world, job, config)
        })?;
        let result = run_rounds(
            model,
            &workers,
            streams,
            random,
            rounds,
            stream_base,
            rollout,
            report,
            #[cfg(test)]
            fail_late,
        );
        // At most one reply per world fits even when a late sampler failure abandons a round.
        let finished = workers.finish();
        result.and(finished)
    })?;
    if rounds == ACTOR_DECISIONS {
        assert!(streams.iter().all(|stream| stream.done));
    }
    #[cfg(test)]
    super::test_support::emit_concurrency_probe_counts(streams, true);
    Ok(())
}

fn operate(
    world: &mut TrainingEnvironment,
    job: Job,
    config: PpoConfig,
) -> Result<Output, PpoError> {
    let Job::Advance {
        state,
        choice,
        space,
        tick,
    } = job
    else {
        return prepare_policy_sample(world).map(|sample| Output::Prepared(Box::new(sample)));
    };
    let mut state = *state;
    let completed = match space {
        Some(space) => advance_cpu(world, &mut state, *choice, *space, config)?,
        None => {
            assert_eq!(choice.action(), crate::StructuredAction::Continue);
            advance_cpu_with_requests(world, &mut state, &choice, config, |world| {
                continue_requests(world, tick)
            })?
        }
    };
    // Queue only CPU state. The GPU owner evaluates retained flushes after full actor success.
    let prepared = if state.done {
        None
    } else {
        Some(prepare_policy_sample(world)?)
    };
    Ok(Output::Advanced(Box::new(Reply {
        state,
        completed,
        prepared,
        opponent: opponent_name(&world.opponent),
    })))
}

fn continue_requests(
    world: &mut TrainingEnvironment,
    tick: u32,
) -> Result<Vec<Option<Request>>, PpoError> {
    let mut requests = Vec::with_capacity(world.seats.len());
    for index in 0..world.seats.len() {
        let seat = &mut world.seats[index];
        if index == world.policy_seat {
            if seat
                .tracker
                .current()
                .is_none_or(|snapshot| snapshot.tick != tick)
            {
                return Err(PpoError::InvalidTransition("prepared actor action space"));
            }
            seat.local
                .note_decision(tick, ActionKind::Continue)
                .map_err(text_error)?;
            // Continue was decoded/validated before dispatch and issues no transport order.
            assert!(
                seat.order_bookkeeping
                    .transport(&seat.persistence)
                    .should_send(None)
                    .is_none()
            );
            requests.push(None);
        } else {
            requests.push(super::super::opponent_request(seat, &mut world.opponent)?);
        }
    }
    Ok(requests)
}

#[allow(clippy::too_many_arguments)]
fn run_rounds(
    model: &PolicyModel,
    workers: &Workers<'_>,
    streams: &mut [EpisodeStream],
    random: &mut [PpoRng],
    rounds: usize,
    stream_base: usize,
    rollout: &mut PpoRollout,
    report: &mut PpoSmokeReport,
    #[cfg(test)] fail_late: bool,
) -> Result<(), PpoError> {
    let active = (0..streams.len()).collect::<Vec<_>>();
    for &stream in &active {
        workers.submit(stream, Job::Prepare)?;
    }
    let mut prepared = workers
        .receive(&active)?
        .into_iter()
        .map(|reply| match reply {
            Output::Prepared(sample) => Some(*sample),
            Output::Advanced(_) => unreachable!("prepare reply"),
        })
        .collect::<Vec<_>>();
    for _ in 0..rounds {
        let active = (0..streams.len())
            .filter(|&stream| !streams[stream].done)
            .collect::<Vec<_>>();
        if active.is_empty() {
            break;
        }
        let (frames, spaces) = active
            .iter()
            .map(|&stream| {
                prepared[stream]
                    .take()
                    .ok_or(PpoError::InvalidTransition("episode prepared frame"))
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .unzip();
        dispatch_round(
            model,
            workers,
            streams,
            random,
            &active,
            frames,
            spaces,
            #[cfg(test)]
            fail_late,
        )?;
        for (stream, (reply, value)) in active
            .iter()
            .copied()
            .zip(receive_round(model, workers, &active)?)
        {
            apply_reply(
                stream_base + stream,
                (*reply, value),
                &mut streams[stream],
                &mut prepared[stream],
                rollout,
                report,
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn dispatch_round(
    model: &PolicyModel,
    workers: &Workers<'_>,
    streams: &mut [EpisodeStream],
    random: &mut [PpoRng],
    active: &[usize],
    frames: Vec<FeatureFrame>,
    spaces: Vec<ActionSpace>,
    #[cfg(test)] fail_late: bool,
) -> Result<(), PpoError> {
    validate_active(random, active)?;
    let mut staged = active
        .iter()
        .map(|&stream| random[stream].clone())
        .collect::<Vec<_>>();
    let mut pending = vec![false; active.len()];
    let sampled = model.sample_batch_continue(
        &frames,
        &spaces,
        &mut staged,
        |row, choice| {
            assert!(!pending[row]);
            let stream = active[row];
            let job = Job::Advance {
                state: Box::new(std::mem::take(&mut streams[stream])),
                choice: Box::new(choice),
                space: None,
                tick: spaces[row].tick(),
            };
            workers
                .submit(stream, job)
                .map_err(|error| crate::ModelError::Backend(error.to_string()))?;
            pending[row] = true;
            Ok(())
        },
        #[cfg(test)]
        fail_late,
    );
    let choices = match sampled {
        Ok(choices) => choices,
        Err(error) => {
            let sent = active
                .iter()
                .copied()
                .zip(&pending)
                .filter_map(|(stream, sent)| sent.then_some(stream))
                .collect::<Vec<_>>();
            let _ = workers.receive(&sent);
            return Err(text_error(error));
        }
    };
    for (&stream, state) in active.iter().zip(staged) {
        random[stream] = state;
    }
    for (row, (choice, space)) in choices.into_iter().zip(spaces).enumerate() {
        if let Some(choice) = choice {
            assert!(!pending[row]);
            let stream = active[row];
            let tick = space.tick();
            workers.submit(
                stream,
                Job::Advance {
                    state: Box::new(std::mem::take(&mut streams[stream])),
                    choice: Box::new(choice),
                    space: Some(Box::new(space)),
                    tick,
                },
            )?;
            pending[row] = true;
        }
    }
    assert!(pending.iter().all(|pending| *pending));
    Ok(())
}

fn receive_round(
    model: &PolicyModel,
    workers: &Workers<'_>,
    active: &[usize],
) -> Result<Vec<ValuedReply>, PpoError> {
    assert!(active.len() <= crate::PPO_ANNEALED_MAX_GAMES);
    let mut replies = Vec::with_capacity(active.len());
    let mut worker_failure = None;
    let mut value_failure = None;
    for &stream in active {
        let reply = match workers.receive(&[stream]) {
            Ok(mut replies) => match replies.pop() {
                Some(Output::Advanced(reply)) => reply,
                _ => {
                    worker_failure.get_or_insert(PpoError::InvalidConfig("Continue overlap reply"));
                    continue;
                }
            },
            Err(error) => {
                worker_failure.get_or_insert(error);
                continue;
            }
        };
        if worker_failure.is_some() || value_failure.is_some() {
            continue;
        }
        // Actor decoding is finished; batch-one flushes may now overlap later CPU replies.
        match flush_value_for_reply(model, &reply) {
            Ok(value) => replies.push((reply, value)),
            Err(error) => {
                value_failure = Some(error);
            }
        }
    }
    // The historical collector drains CPU errors before consuming any flush result.
    if let Some(error) = worker_failure.or(value_failure) {
        return Err(error);
    }
    assert_eq!(replies.len(), active.len());
    Ok(replies)
}

fn flush_value_for_reply(model: &PolicyModel, reply: &Reply) -> Result<Option<f32>, PpoError> {
    if !reply.state.should_flush() || reply.state.done {
        return Ok(None);
    }
    let frame = &reply
        .prepared
        .as_ref()
        .ok_or(PpoError::InvalidTransition("episode prepared frame"))?
        .0;
    Ok(Some(
        model
            .evaluate_batch(std::slice::from_ref(frame))
            .map_err(text_error)?[0]
            .value,
    ))
}

fn apply_reply(
    stream: usize,
    ready: (Reply, Option<f32>),
    state: &mut EpisodeStream,
    prepared: &mut Option<(FeatureFrame, ActionSpace)>,
    rollout: &mut PpoRollout,
    report: &mut PpoSmokeReport,
) -> Result<(), PpoError> {
    let (reply, value) = ready;
    let Reply {
        state: next,
        completed,
        prepared: sample,
        opponent,
    } = reply;
    *state = next;
    *prepared = sample;
    finish_advance_from_parts(state, stream, completed, value, opponent, rollout, report)
}
