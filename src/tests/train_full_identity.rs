//! Default train-full scope and self-contained CUDA campaign goldens.

fn canonical_settings() -> crate::TrainingJobConfig {
    // Replay the recorded invocation rather than inheriting changing CLI defaults.
    let arguments = concat!(
        "--complete-episodes --environments 2 --rollout 1163 --epochs 1 --minibatch 512 ",
        "--map 2 --seed 10141700 --gamma-per-tick 1 --gae-lambda .95 ",
        "--learning-rate .00003 --entropy-coefficient .001 --opponent-schedule mastery-v1 ",
        "--mastery-window 50 --mastery-win-percent 80",
    )
    .split_ascii_whitespace()
    .collect::<Vec<_>>();
    crate::cli::training_settings_for_test(&arguments).expect("canonical train-full settings")
}

#[test]
fn canonical_train_full_scope_is_pinned() {
    let settings = canonical_settings();
    let run = crate::ppo_arena::training_checkpoint_run(
        &settings,
        crate::PolicyDevice::Cpu,
        settings.ppo,
    )
    .expect("canonical run scope");
    assert_eq!(
        run.command_line,
        "train-full --environments 2 --rollout 1163 --epochs 1 --minibatch 512 --seed 10141700 --map 2 --device cpu --complete-episodes --opponent-schedule mastery-v1 --mastery-window 50 --opponent-win-percent weak=80 --opponent-win-percent teacher=80"
    );
    assert_eq!(run.run_seed, 10_141_700);
    assert_eq!(run.map, bota_proto::MapId(2));
    assert_eq!(run.hero, crate::SHADOW_FIEND);
    assert_eq!(run.batch_size, 512);
    assert_eq!(run.rules_audit_version, crate::PPO_RULES_AUDIT_VERSION);
    assert!(run.mastery_config.is_some());
    assert_eq!(run.enabled_features, crate::compiled_features());
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
mod cuda {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::fmt::Write;

    // Successful weights/Adam+weights records from pipeline-groups-20260917/ab-results.jsonl.
    const GOLDENS: [(usize, &str, &str, &str); 3] = [
        (
            2,
            "e02g1u1",
            "fd8bf5bd21968263be8f413f6113b3600eb2a37de216ea97d33fbde38889ee98",
            "0e31970182db99fafc6a5400cf00332bb27505926aa480063ba0a472c05bdce0",
        ),
        (
            8,
            "e08g1u1",
            "c5088933dab8fe2269b96408cf55030d70aad108dd7b377e4b8d10a8b3c85d43",
            "0a83ff5641643a1f448126b6c28725884b670094b926ebb403e8db400b3d655a",
        ),
        (
            16,
            "e16g1u1",
            "b33ea6f7cde6cbae1d026057e770704d64216b155fb65bcc0db43572733d35c1",
            "91aa5af0263685372a62ee1858a9e65700327096d7101067e096a14c88c7af74",
        ),
    ];

    #[test]
    fn digest_preserves_lowercase_hex_and_leading_zeroes() {
        assert_eq!(
            digest(&[]),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            digest(&[&[0.0]]),
            "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
        );
    }

    fn digest(parts: &[&[f32]]) -> String {
        let mut hash = Sha256::new();
        for values in parts {
            for value in *values {
                hash.update(value.to_le_bytes());
            }
        }
        let bytes = hash.finalize();
        let mut hex = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            write!(hex, "{byte:02x}").expect("write digest hex to String");
        }
        hex
    }

    #[test]
    #[ignore = "requires the owner's exclusive CPU/CUDA runner; full recorded episodes"]
    fn recorded_identity_replays_three_goldens_and_default_cpu_cuda_parity() {
        for (environments, tag, weights, optimizer) in GOLDENS {
            let cuda = replay(environments, crate::PolicyDevice::Cuda { ordinal: 0 });
            let (first, second) = cuda.snapshot.adam.moments();
            assert_eq!(
                digest(&[&cuda.snapshot.parameters]),
                weights,
                "{tag} weights"
            );
            assert_eq!(
                digest(&[first, second, &cuda.snapshot.parameters]),
                optimizer,
                "{tag} Adam"
            );
            if environments == 2 {
                let cpu = replay(environments, crate::PolicyDevice::Cpu);
                assert_eq!(cpu.progress, cuda.progress);
                assert_eq!(cpu.rng, cuda.rng);
                assert_eq!(
                    cpu.snapshot.parameters.len(),
                    cuda.snapshot.parameters.len()
                );
                for (cpu, cuda) in cpu
                    .snapshot
                    .parameters
                    .iter()
                    .zip(&cuda.snapshot.parameters)
                {
                    assert!((cpu - cuda).abs() <= 1.0e-5, "CPU={cpu} CUDA={cuda}");
                }
            }
        }
    }

    struct Replay {
        progress: crate::CheckpointProgress,
        rng: (u64, u64),
        snapshot: crate::ModelAdamSnapshot,
    }

    fn replay(environments: usize, device: crate::PolicyDevice) -> Replay {
        let directory = super::super::map2_checkpoint::Directory::new();
        let mut settings = canonical_settings();
        settings.ppo.environments = environments;
        settings.updates = 1;
        crate::run_training_job_on_with_initial_weights(
            settings,
            device,
            &directory.0,
            false,
            None,
            |_| {},
        )
        .expect("identity update");
        let artifact = crate::TrainingArtifact::load(&directory.0).expect("artifact");
        let model = crate::PolicyModel::fresh_on(1, device).expect("model");
        let state = artifact.restore(&model, artifact.run()).expect("restore");
        Replay {
            progress: artifact.progress().clone(),
            rng: state.trainer().rng_checkpoint(),
            snapshot: state
                .trainer()
                .checkpoint_snapshot(&model)
                .expect("snapshot"),
        }
    }
}
