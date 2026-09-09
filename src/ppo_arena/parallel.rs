use crate::PpoError;

pub(super) fn ordered_active<T: Send, I: Send, O: Send>(
    worlds: &mut [T],
    jobs: Vec<(usize, I)>,
    operation: impl Fn(usize, &mut T, I) -> Result<O, PpoError> + Sync,
) -> Result<Vec<O>, PpoError> {
    if worlds.len() > super::TRAINING_MAX_ENVIRONMENTS
        || jobs.iter().any(|(stream, _)| *stream >= worlds.len())
        || jobs.windows(2).any(|pair| pair[0].0 >= pair[1].0)
    {
        return Err(PpoError::InvalidConfig("parallel episode jobs"));
    }
    assert!(jobs.len() <= worlds.len());
    assert!(jobs.len() <= super::TRAINING_MAX_ENVIRONMENTS);
    if jobs.len() <= 1 {
        return jobs
            .into_iter()
            .map(|(stream, input)| operation(stream, &mut worlds[stream], input))
            .collect();
    }
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(jobs.len());
        let mut pending = jobs.into_iter().peekable();
        let mut failure = None;
        for (stream, world) in worlds.iter_mut().enumerate() {
            if pending.peek().is_none_or(|job| job.0 != stream) {
                continue;
            }
            let (_, input) = pending.next().expect("matching job");
            let operation = &operation;
            match std::thread::Builder::new()
                .name(format!("ppo-episode-{stream}"))
                .spawn_scoped(scope, move || operation(stream, world, input))
            {
                Ok(handle) => handles.push((stream, handle)),
                Err(error) => {
                    failure = Some(PpoError::EpisodeWorker {
                        stream,
                        cause: error.to_string(),
                    });
                    break;
                }
            }
        }
        join_ordered(handles, failure)
    })
}

fn join_ordered<O>(
    handles: Vec<(
        usize,
        std::thread::ScopedJoinHandle<'_, Result<O, PpoError>>,
    )>,
    mut failure: Option<PpoError>,
) -> Result<Vec<O>, PpoError> {
    assert!(handles.len() <= super::TRAINING_MAX_ENVIRONMENTS);
    let mut output = Vec::with_capacity(handles.len());
    for (stream, handle) in handles {
        let result = handle.join().unwrap_or_else(|_| {
            Err(PpoError::EpisodeWorker {
                stream,
                cause: "thread panicked".to_owned(),
            })
        });
        match result {
            Ok(value) => output.push(value),
            Err(error) => {
                failure.get_or_insert(error);
            }
        }
    }
    assert!(output.len() <= super::TRAINING_MAX_ENVIRONMENTS);
    failure.map_or(Ok(output), Err)
}

#[cfg(test)]
#[path = "../tests/episode_parallel.rs"]
mod tests;
