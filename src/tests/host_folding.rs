use super::*;

#[test]
fn coordinate_partitions_preserve_fold_bits_for_one_to_thirty_two_workers() {
    for workers in [1, 2, 8, 16, 32] {
        let mut source = (0..257)
            .map(|index| (index as f32 - 128.0) * 0.001)
            .collect::<Vec<_>>();
        source[0] = -0.0;
        source[1] = f32::from_bits(1);
        source[2] = f32::from_bits(0x8000_0001);
        let mut expected = vec![0.0f32; source.len()];
        std::thread::scope(|scope| {
            let mut pool = FoldPool::spawn(scope, source.len(), workers, 3).expect("pool");
            for scale in [64.0f32, 64.0, 1.0] {
                let mut addition = source.clone();
                validate_gradients(&addition).expect("finite");
                scale_gradients(&mut addition, scale).expect("scale");
                accumulate_gradients(&mut expected, &addition).expect("serial addition");
                pool.submit(source.clone(), scale).expect("submit");
                pool.receive().expect("fold");
            }
            let actual = pool.finish().expect("finish");
            assert_eq!(actual.len(), expected.len());
            assert!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        });
    }
}

#[test]
fn fold_error_uses_serial_phase_then_global_coordinate_order() {
    std::thread::scope(|scope| {
        let mut pool = FoldPool::spawn(scope, 8, 2, 1).expect("pool");
        let mut values = vec![1.0; 8];
        values[0] = f32::MAX;
        values[6] = f32::NAN;
        pool.submit(values, 64.0).expect("submit");
        assert_eq!(
            pool.receive()
                .expect_err("validation before scale")
                .to_string(),
            "model gradient 6 is non-finite"
        );
    });
}

#[test]
fn host_worker_resolution_is_local_bounded_and_adaptive() {
    assert_eq!(worker_count(32, 8, MODEL_PARAMETER_COUNT), 8);
    assert_eq!(worker_count(32, 32, MODEL_PARAMETER_COUNT), 32);
    assert_eq!(worker_count(32, 32, 16), 1);
    assert_eq!(worker_count(1, 32, MODEL_PARAMETER_COUNT), 1);
}
