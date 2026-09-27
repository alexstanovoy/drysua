//! Shared filesystem and runtime fixtures; behavior lives in the public checkpoint tests.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use safetensors::tensor::{Dtype, TensorView, serialize};

pub(super) struct Directory(pub PathBuf);

impl Directory {
    pub(super) fn new() -> Self {
        static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "drysua-map2-checkpoint-{}-{}",
            std::process::id(),
            NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).expect("unique fixture directory");
        Self(path)
    }
}

impl Drop for Directory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("fixture cleanup");
    }
}

pub(super) fn runtime_bytes(values: &[f32], metadata: HashMap<String, String>) -> Vec<u8> {
    let data: Vec<_> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let tensor = TensorView::new(Dtype::F32, vec![values.len()], &data).expect("tensor");
    serialize([("model.parameters", tensor)], Some(metadata)).expect("fixture")
}

pub(super) fn config() -> crate::PpoConfig {
    crate::PpoConfig {
        gamma_tick: crate::MAP2_REWARD_GAMMA_TICK,
        rollout_decisions: 2,
        environments: 2,
        minibatch: 2,
        ..crate::PpoConfig::default()
    }
}
