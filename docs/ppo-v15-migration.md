# PPO v15: normalized, purchase-neutral, episodically bounded rewards

PPO schema v15/hash `13893101989595893928`, learner rules audit v14, and
league schema v16/hash `1035319045739487525` supersede PPO v14 for training.

The reward repair removes cash shaping entirely. Unspent cash is not total
wealth: buying useful equipment must not produce a negative shaping reward.
The retained potential coefficients are XP advantage `.02/101`, combat `2/101`,
and structure advantage `5/101`. Every component uses the same normalization;
terminal outcomes are now win `+1`, loss `-1`, draw `0`.

All terminal outcomes use zero next potential, including draws. The terminal
shaping term is consequently `-previous_potential`, independent of the final
snapshot. Truncation is not terminal and retains the usual discounted next
potential and value bootstrap.

The episode budget is `100/101` for the sum of **absolute emitted shaping
components**, not signed cumulative reward. Opposite-sign rewards cannot refill
it, nor can opposite-sign components cancel expenditure. When the remaining
budget is insufficient, all shaping components are scaled proportionally.
The expenditure accumulator uses f64; emitted rewards remain f32 (roundoff
tolerance `1e-6` in bound tests). An episode's total absolute shaping is below
one, and a transition's absolute total reward is at most `1 + 100/101` up to
floating-point roundoff. Terminal dominance here is an **undiscounted** budget
statement, not a claim about arbitrarily distant discounted outcomes.

Clipping a potential-based reward to a finite budget is not a policy-invariance
theorem. The bounded budget is an explicit safety tradeoff. Widening public
scoreboard subtraction before conversion also prevents i64 overflow or u64
sign-wrap from manufacturing invalid potentials at the numeric boundary.

## Deliberately unchanged

Permanent critic-input detachment, LR `3e-6`, and the normalized reward scale are
explicit **conservative design/tuning decisions** for BC transfer. Shared-trunk
actor/critic gradients are standard and not inherently incorrect. The isolation
test proves the chosen no-critic-drift property, not that shared representations
are invalid or that the chosen settings improve learned strength. The reused-dev
experiment did not establish an improvement over the accepted BC anchor.

- Gamma `.9966555` per simulation tick and GAE lambda `.98`. Gamma's e-folding
  horizon is approximately 298.5 ticks, or 10 seconds at 30 Hz. With decisions
  every three ticks, the GAE trace horizon is only about 3.3 seconds. This is
  short relative to a structure win; retain it for the first repaired-reward
  run rather than confound reward correctness with a horizon sweep.
- PPO LR `3e-6`, value coefficient `.5`, entropy `.01`, global clip `.5`,
  detached value input, and post-step KL rollback from v14. Reward normalization
  reduces target scale but does not guarantee balanced actor/critic gradients;
  an untrained value head can still dominate their shared gradient clip.
- Model architecture, inference, BC, Teacher, and action/feature/model schemas.

## Rollback boundary and regression coverage

Post-step candidate KL is weighted by rows across the entire effective minibatch,
including an uneven final microbatch. Release regression tests cover 65 and 81
rows and independently compare the rejected candidate's KL with the weighted
likelihood of the same retained candidate. They distinguish row weighting from
incorrect equal weighting of microbatches.

Test-only per-call hooks also inject a candidate-evaluation error after the first
64 rows and a failure during rollback parameter import. When rollback succeeds,
tests verify exact parameters, nonzero Adam moments, Adam step, optimizer binding,
and model revision restoration. The original candidate error is propagated.

If rollback import itself fails, the returned error preserves both the candidate
evaluation/rejection cause and rollback failure context. **Atomic restoration is
not guaranteed when the backend fails during rollback.** Callers receive an error,
not an accepted update or a promise that the pre-step state was restored. The
injected import failure uses the existing parameter-replacement failure hook; it
does not simulate every possible backend or device failure.

This finalization changes error diagnostics and test coverage, not the numerical
update/reward algorithm, parameter layout, or training schema.

## Compatibility

Durable training checkpoints from PPO v13 **or v14** must not resume: reward
targets and value scale changed. Both strict and provenance-compatible loads
still validate the current PPO version/hash. No metadata rewriting is allowed.

The existing exact audited v13 runtime tuple remains accepted, allowing the
original BC anchor to initialize a **fresh** v15 optimizer. The loader still
checks every metadata field and tensor name/shape/dtype/finite value. v14 runtime
compatibility is not added because this experiment uses the original v13 BC
anchor; v14 and other unlisted runtime tuples are rejected. No accepted artifact
is modified. New runtime weights carry v15 metadata.

Training/evaluation evidence belongs under `artifacts/temp`; the schema repair
does not itself establish stronger gameplay or authorize neural promotion.
