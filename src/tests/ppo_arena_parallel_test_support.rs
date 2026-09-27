use super::*;

pub(crate) fn assert_worker_capacity_for_test() {
    for count in [0, 41] {
        let mut worlds = vec![0; count];
        std::thread::scope(|scope| {
            assert_eq!(
                StreamWorkers::spawn(scope, &mut worlds, "ppo", |_, _, ()| Ok(())).err(),
                Some(PpoError::InvalidConfig("stream worker environments"))
            );
        });
        assert!(worlds.iter().all(|world| *world == 0));
    }
}

#[cfg(test)]
#[path = "../tests/episode_parallel.rs"]
mod tests;
