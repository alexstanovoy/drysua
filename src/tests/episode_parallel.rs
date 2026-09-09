use std::sync::{
    Barrier,
    atomic::{AtomicUsize, Ordering},
};

use super::*;

#[test]
fn active_jobs_run_concurrently_and_return_in_stream_order() {
    let mut worlds = [10, 20, 30, 40];
    let barrier = Barrier::new(2);
    let results = ordered_active(&mut worlds, vec![(1, 7), (3, 9)], |stream, world, value| {
        barrier.wait();
        *world += value;
        Ok((stream, *world))
    })
    .expect("parallel jobs");

    assert_eq!(results, [(1, 27), (3, 49)]);
    assert_eq!(worlds, [10, 27, 30, 49]);
}

#[test]
fn invalid_job_indices_fail_before_any_world_is_modified() {
    for jobs in [
        vec![(1, ()), (1, ())],
        vec![(2, ()), (0, ())],
        vec![(3, ())],
    ] {
        let mut worlds = [0; 3];
        let error = ordered_active(&mut worlds, jobs, |_, world, _| {
            *world += 1;
            Ok(())
        })
        .expect_err("invalid schedule");

        assert_eq!(
            error.to_string(),
            "invalid PPO config field: parallel episode jobs"
        );
        assert_eq!(worlds, [0; 3]);
    }
}

#[test]
fn worker_errors_do_not_abandon_other_jobs() {
    let mut worlds = [0; 3];
    let completed = AtomicUsize::new(0);
    let error = ordered_active(
        &mut worlds,
        vec![(0, ()), (1, ()), (2, ())],
        |stream, world, _| {
            *world = 1;
            completed.fetch_add(1, Ordering::Relaxed);
            if stream == 0 {
                Err(PpoError::NonFinite("test job"))
            } else {
                Ok(())
            }
        },
    )
    .expect_err("worker error");

    assert_eq!(error, PpoError::NonFinite("test job"));
    assert_eq!(completed.load(Ordering::Relaxed), 3);
    assert_eq!(worlds, [1; 3]);
}

#[test]
fn worker_panics_are_reported_after_other_workers_are_joined() {
    let mut worlds = [0; 3];
    let error = ordered_active(
        &mut worlds,
        vec![(0, ()), (1, ()), (2, ())],
        |stream, world, _| {
            assert_ne!(stream, 0, "injected worker panic");
            *world = 1;
            Ok(())
        },
    )
    .expect_err("worker panic");

    assert_eq!(
        error.to_string(),
        "PPO episode worker 0 failed: thread panicked"
    );
    assert_eq!(worlds, [0, 1, 1]);
}

#[test]
fn empty_and_single_jobs_need_no_parallel_barrier() {
    let mut worlds = [0];
    let empty: Vec<()> = ordered_active(&mut worlds, vec![], |_, _, ()| Ok(())).expect("empty");
    let single = ordered_active(&mut worlds, vec![(0, 7)], |_, world, value| {
        *world = value;
        Ok(value)
    })
    .expect("single");

    assert!(empty.is_empty());
    assert_eq!(single, [7]);
    assert_eq!(worlds, [7]);
}
