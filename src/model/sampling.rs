//! Skip whole unused actor heads, never rows of an exercised GEMM: row compaction
//! can change floating-point results and therefore sampled trajectories.

use super::*;

pub(super) fn kind_needed(prefixes: &[TrainingPrefix]) -> [bool; 2] {
    needed(prefixes, |kind| {
        [
            !matches!(kind, ActionKind::Continue | ActionKind::Learn),
            kind == ActionKind::Learn,
        ]
    })
}

pub(super) fn needed<const HEADS: usize>(
    prefixes: &[TrainingPrefix],
    demand: impl Fn(ActionKind) -> [bool; HEADS],
) -> [bool; HEADS] {
    assert!(!prefixes.is_empty());
    assert!(prefixes.len() <= MODEL_SAMPLING_BATCH);
    #[cfg(test)]
    if EAGER.get() {
        return [true; HEADS];
    }
    let mut needed = [false; HEADS];
    for prefix in prefixes {
        for (needed, required) in needed.iter_mut().zip(demand(prefix.kind())) {
            *needed |= required;
        }
    }
    needed
}

#[cfg(test)]
thread_local! {
    static EAGER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static DISPATCHES: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
}

#[cfg(test)]
pub(super) fn record_dispatch(batch: usize) {
    assert!(batch > 0);
    assert!(batch <= MODEL_SAMPLING_BATCH);
    let (heads, rows) = DISPATCHES.get();
    DISPATCHES.set((
        heads.checked_add(1).expect("bounded test dispatches"),
        rows.checked_add(batch).expect("bounded test rows"),
    ));
}

#[cfg(test)]
pub(crate) fn take_sampling_dispatches_for_test() -> (usize, usize) {
    DISPATCHES.replace((0, 0))
}

#[cfg(test)]
pub(crate) fn with_eager_sampling_for_test<T>(operation: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            EAGER.set(self.0);
        }
    }
    let _reset = Reset(EAGER.replace(true));
    operation()
}
