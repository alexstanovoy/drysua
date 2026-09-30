use super::*;

pub(super) fn contract() -> Value {
    json!({"schema": SCHEMA, "kind": "contract", "model": model(), "enabled_features": compiled_features(),
        "capabilities": {"inspection": cfg!(target_os = "linux"), "annealed_history": cfg!(feature = "builtin"),
            "read_only": true, "strict_build_features": true, "controller_run_kind": "train-annealed"},
        "limits": {"max_json_bytes": MAX_JSON_BYTES, "max_snapshots": MAX_SNAPSHOTS, "max_files": MAX_FILES,
            "manifest_bytes": MAX_META_BYTES, "training_tensor_bytes": MAX_TRAINING_TENSOR_BYTES,
            "runtime_tensor_bytes": MAX_RUNTIME_TENSOR_BYTES, "snapshot_bytes": 4096,
            "max_samples": crate::PPO_MAX_SAMPLES, "max_games": crate::PPO_MAX_GAMES},
        "schemas": {"action": schema(ACTION_SCHEMA_VERSION, ACTION_SCHEMA_HASH),
            "feature": schema(FEATURE_SCHEMA_VERSION, FEATURE_SCHEMA_HASH),
            "reward": schema(crate::MAP2_REWARD_SCHEMA_VERSION, crate::MAP2_REWARD_SCHEMA_HASH),
            "ppo": schema(PPO_SCHEMA_VERSION, PPO_SCHEMA_HASH),
            "rules_audit_version": PPO_RULES_AUDIT_VERSION},
        "checkpoint": checkpoint(), "numeric_semantics": {"ppo_floats": "IEEE-754 binary32, JSON numbers widened exactly to binary64",
            "adaptive_rate_units": 1_000_000, "schema_hash": "lowercase hexadecimal FNV-1a u64 (16 digits)",
            "file_hash": "lowercase SHA-256 (64 digits)"}})
}

pub(super) fn checkpoint() -> Value {
    schema(CHECKPOINT_SCHEMA_VERSION, CHECKPOINT_SCHEMA_HASH)
}

fn schema(version: u32, hash: u64) -> Value {
    json!({"version": version, "hash": format!("{hash:016x}")})
}

pub(super) fn model() -> Value {
    json!({"version": MODEL_SCHEMA_VERSION, "hash": format!("{MODEL_SCHEMA_HASH:016x}"), "parameters": MODEL_PARAMETER_COUNT})
}

pub(super) fn run(run: &CheckpointRun) -> Value {
    let device = match run.device {
        CheckpointDevice::Cpu => json!({"kind": "cpu", "ordinal": null}),
        CheckpointDevice::Cuda { ordinal } => json!({"kind": "cuda", "ordinal": ordinal}),
    };
    json!({"git_commit": run.git_commit, "simulator_commit": run.simulator_commit,
        "enabled_features": run.enabled_features, "command_line": run.command_line, "run_seed": run.run_seed,
        "map": run.map.0, "hero": run.hero.0, "device": device, "batch_size": run.batch_size,
        "rules_audit_version": run.rules_audit_version})
}

pub(super) fn progress(artifact: &TrainingArtifact, games: Option<u64>) -> Value {
    let progress = &artifact.progress;
    let random = progress
        .rng_states
        .iter()
        .map(|rng| json!({"name": rng.name(), "state": rng.state(), "draws": rng.draws()}))
        .collect::<Vec<_>>();
    assert!(random.len() <= MAX_RNG_STATES);
    json!({"updates": progress.global_update, "optimizer_steps": artifact.optimizer.step,
        "rollout_samples": progress.rollout_samples, "games": games, "policy_version": progress.policy_version,
        "scheduler_step": progress.scheduler_step, "curriculum_stage": progress.curriculum_stage,
        "best_evaluation": progress.best_evaluation, "rng_states": random,
        "shuffle_rng": {"state": artifact.shuffle.0, "draws": artifact.shuffle.1},
        "league_references": progress.league_references})
}

pub(super) fn ppo(config: PpoConfig) -> Value {
    json!({"schema_version": PPO_SCHEMA_VERSION, "schema_hash": format!("{PPO_SCHEMA_HASH:016x}"),
        "decision_interval_ticks": config.decision_interval_ticks, "rollout_decisions": config.rollout_decisions,
        "environments": config.environments, "epochs": config.epochs, "minibatch": config.minibatch,
        "clip_epsilon": f64::from(config.clip_epsilon), "value_coefficient": f64::from(config.value_coefficient),
        "entropy_coefficient": f64::from(config.entropy_coefficient), "learning_rate": f64::from(config.learning_rate),
        "adam_beta1": f64::from(config.adam_beta1), "adam_beta2": f64::from(config.adam_beta2),
        "adam_epsilon": f64::from(config.adam_epsilon), "gradient_clip": f64::from(config.gradient_clip),
        "gamma_tick": f64::from(config.gamma_tick), "gae_lambda": f64::from(config.gae_lambda),
        "target_kl": f64::from(config.target_kl),
        "f32_bits": {"clip_epsilon": config.clip_epsilon.to_bits(), "value_coefficient": config.value_coefficient.to_bits(),
            "entropy_coefficient": config.entropy_coefficient.to_bits(), "learning_rate": config.learning_rate.to_bits(),
            "adam_beta1": config.adam_beta1.to_bits(), "adam_beta2": config.adam_beta2.to_bits(),
            "adam_epsilon": config.adam_epsilon.to_bits(), "gradient_clip": config.gradient_clip.to_bits(),
            "gamma_tick": config.gamma_tick.to_bits(), "gae_lambda": config.gae_lambda.to_bits(), "target_kl": config.target_kl.to_bits()}})
}

pub(super) fn adaptive(progress: &CheckpointProgress) -> Value {
    let Some(checkpoint) = progress.adaptive_environment else {
        return Value::Null;
    };
    json!({"config": {"success_updates": checkpoint.config.success_updates,
            "success_rate_units": checkpoint.config.success_rate.units(), "poor_updates": checkpoint.config.poor_updates,
            "poor_rate_units": checkpoint.config.poor_rate.units(), "extension_units": checkpoint.config.extension.units()},
        "limits": {"base_updates": checkpoint.limits.base_updates, "total_updates": checkpoint.limits.total_updates,
            "zero_updates": checkpoint.limits.zero_updates},
        "state": {"generation": checkpoint.state.generation, "start_update": checkpoint.state.start_update,
            "updates_in_generation": checkpoint.state.updates_in_generation, "success_streak": checkpoint.state.success_streak,
            "poor_streak": checkpoint.state.poor_streak, "extension_awards": checkpoint.state.extension_awards},
        "snapshot_count": checkpoint.snapshot_count, "snapshot_hash": hex(&checkpoint.snapshot_hash)})
}
