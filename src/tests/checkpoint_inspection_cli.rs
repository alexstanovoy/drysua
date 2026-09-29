use super::*;

#[test]
fn checkpoint_inspect_accepts_exactly_one_read_only_input_mode() {
    for mode in [
        vec!["--contract"],
        vec!["--checkpoint-directory", "unused-directory"],
    ] {
        let mut args = vec!["drysua", "checkpoint-inspect"];
        args.extend(mode);
        let parsed = Cli::try_parse_from(args).expect("inspection command");
        assert!(matches!(
            parsed.operation,
            Some(Operation::CheckpointInspect(_))
        ));
    }
}

#[test]
fn checkpoint_inspect_rejects_missing_conflicting_and_mutating_options() {
    for options in [
        vec![],
        vec!["--contract", "--checkpoint-directory", "unused-directory"],
        vec!["--contract", "--resume"],
        vec!["--contract", "--device", "cuda"],
        vec![
            "--checkpoint-directory",
            "one",
            "--checkpoint-directory",
            "two",
        ],
    ] {
        let mut args = vec!["drysua", "checkpoint-inspect"];
        args.extend(options);
        let error = Cli::try_parse_from(args)
            .err()
            .expect("closed inspection interface");
        assert!(!matches!(
            error.kind(),
            clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
        ));
        assert!(error.to_string().contains("error:"));
    }
}
