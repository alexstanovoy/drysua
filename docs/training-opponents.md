# Full-episode Map2 opponent schedule

`train-full --opponent-schedule teacher` is the default and retains the original
Teacher-only full-episode behavior. The existing reset-window collector is unchanged;
nondefault opponent schedules require `--complete-episodes`.

`--opponent-schedule weak-warmup-v1` is an explicit deterministic curriculum:

- Global updates **0 and1** use only the existing Weak opponent (Continue-only).
- Starting at update **2**, pair `p` uses Teacher when `(update - 2 + p) % 3 == 0`,
  and Weak otherwise. Pair indices start at0; both learner seats of a pair face
  the same opponent with the same existing derived arena seed.
- E6 therefore uses4Weak/2Teacher per update, rotating the Teacher pair. With E4,
  the three-update cycle is2Weak/2Teacher,4Weak,2Weak/2Teacher. E2 cycles one Teacher
  pair then two Weak pairs. No new RNG draws, seed-dependent difficulty, performance
  threshold, live-policy fallback or simulator-private policy input is introduced.

The schedule is **training run configuration**, not a model/feature/action/reward
change: F17/M19/A5/PPO32/reward3 and runtime weights remain unchanged. Its versioned
identity is added to the canonical `CheckpointRun.command_line` for nondefault
schedules. Strict checkpoint compatibility already compares this before reading
tensor payloads or installing parameters/optimizer state, so changing the schedule
cannot silently resume. Explicit/default Teacher use the same legacy command,
without a new suffix. Git provenance compatibility remains strict independently.

Phase is derived from the persisted `global_update`, not invocation-local progress.
The generic reserved `curriculum_stage` field stays0; no duplicate phase counter is
needed. A future schedule with different semantics must have a new versioned name,
not reinterpret `weak-warmup-v1` checkpoints. To compare schedules experimentally,
initialize two fresh trainers from the same runtime weights; do not compare a fresh
curriculum optimizer against a control with inherited Adam moments.

Collection telemetry reports schedule, global update, actual Weak/Teacher counts
and batch label. Each episode and reward record names its actual runtime opponent.
Weak victories are not evidence of improved play against Teacher or predecessors;
assess candidates on the same unchanged opponents and paired evaluation seeds.

## Initial short pilot

The first controlled pilot started both arms from identical M19/u162 weights with
fresh optimizers and ran eight updates per arm. Baseline, Teacher-only control and
curriculum each lost all four matched DEV games against the same frozen Teacher.
Their mean final last-hit counts were 13, 4.5 and 11 respectively. Equal updates
produced different sample/optimizer-step counts because episode lengths differed.

The curriculum recorded 30 wins, one loss and five draws against Weak, but lost
all 12 training episodes against Teacher. This does not establish improved play
against active opponents or fix late first-wave arrival, which was not measured.
Keep the schedule optional and experimental; neither the model nor the schedule
is promoted by this result. Full local evidence is in
`artifacts/temp/m19-curriculum-pilot-20260914/REPORT.md`.
