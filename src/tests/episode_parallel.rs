use std::sync::Barrier;

use super::*;

#[test]
fn persistent_workers_preserve_sparse_order_and_recover_after_errors() {
    let mut worlds = [0usize; 40];
    let barrier = Barrier::new(2);
    std::thread::scope(|scope| {
        let barrier = &barrier;
        let workers =
            StreamWorkers::spawn(scope, &mut worlds, "ppo", move |stream, world, request| {
                match request {
                    0 => return Err(PpoError::InvalidConfig("worker failure")),
                    1 => {
                        barrier.wait();
                    }
                    _ => {}
                }
                *world += request;
                Ok((stream, *world))
            })
            .expect("forty workers");
        for stream in [1, 39] {
            workers.submit(stream, 1).expect("concurrent requests");
        }
        assert_eq!(
            workers.receive(&[1, 39]).expect("ordered replies"),
            [(1, 1), (39, 1)]
        );
        workers.submit(0, 0).expect("injected error");
        workers.submit(39, 2).expect("successful peer");
        assert_eq!(
            workers.receive(&[0, 39]),
            Err(PpoError::InvalidConfig("worker failure"))
        );
        for stream in [0, 39] {
            workers.submit(stream, 2).expect("recovery requests");
        }
        assert_eq!(
            workers.receive(&[0, 39]).expect("drained and recovered"),
            [(0, 2), (39, 5)]
        );
        assert_eq!(
            workers.submit(40, 2),
            Err(PpoError::InvalidConfig("stream worker index"))
        );
        workers.finish().expect("join");
    });
    assert_eq!(worlds[0], 2);
    assert_eq!(worlds[39], 5);
    assert!(worlds[2..39].iter().all(|world| *world == 0));
    super::super::assert_worker_capacity_for_test();
}

#[test]
fn persistent_worker_panic_surfaces_while_other_replies_remain_drainable() {
    let mut worlds = [0; 2];
    std::thread::scope(|scope| {
        let workers = StreamWorkers::spawn(scope, &mut worlds, "ppo", |stream, world, ()| {
            assert_ne!(stream, 0, "injected worker panic");
            *world = 1;
            Ok(())
        })
        .expect("workers");
        for stream in [0, 1] {
            workers.submit(stream, ()).expect("submit");
        }
        assert_eq!(
            workers.receive(&[0]).expect_err("worker panic").to_string(),
            "PPO episode worker 0 failed: stream worker stopped"
        );
        assert_eq!(workers.receive(&[1]).expect("drain peer"), [()]);
        workers.finish().expect("join all workers");
    });
    assert_eq!(worlds, [0, 1]);
}
