use super::*;

#[test]
fn provenance_pool_reuses_buffers_in_the_steady_state() {
    let pool = Arc::new(ProvenancePool::new());
    // Warm up to the number of captures a live history can hold at once.
    let mut live: Vec<PooledProvenance> = (0..PROVENANCE_POOL_LIMIT).map(|_| pool.take()).collect();
    for handle in live.drain(..) {
        drop(handle);
    }
    let warmed = pool.allocations.load(Ordering::Relaxed);
    assert!(warmed > 0, "warm-up must allocate the bounded pool");
    assert!(pool.free.lock().expect("pool lock").len() == PROVENANCE_POOL_LIMIT);
    for _ in 0..10_000 {
        let handle = pool.take();
        drop(handle);
    }
    assert_eq!(
        pool.allocations.load(Ordering::Relaxed),
        warmed,
        "steady-state provenance captures must reuse pooled buffers"
    );
}
