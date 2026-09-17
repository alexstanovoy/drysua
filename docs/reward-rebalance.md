# Map2 reward6: reward5 dense terms with smaller terminal values

These coefficients are engineering choices, not a trained-model strength claim.
Reward6 changes only terminal values. Reward5 dense coefficients and rolling mastery
scheduling/window semantics are unchanged. Historical reward5 reports remain unchanged.

## Diminishing event channels

For magnitude budget B, scale S, previous observed amount C and new amount a,
the magnitude is `B*S*a / ((S+C)*(S+C+a))`, with the sign below. These are total
channel budgets, NOT flat per-hit coefficients. Every channel has independent,
nonreplenishing accounting; input remaining fractions are `S/(S+C)`.

| Channel | Signed B | S |
|---|---:|---:|
| Own paid gold | +0.03 | 300 |
| Enemy paid gold | -0.02 | 300 |
| Own XP | +0.03 | 3000 |
| Enemy XP | -0.02 | 3000 |
| Own hero damage to enemy hero | +0.08 | 1600 |
| Incoming enemy-hero damage | -0.05 | 1600 |
| Incoming lane OR neutral creep damage | -0.1 | 1600 |
| Other/unknown/environmental received damage | -0.005 | 500 |
| Observed mana expenditure | -0.04 | 1200 |
| NEW incoming known Tower damage | -0.1 | 500 |

Known Towers of any team use only the new Tower channel and raw counter, never
Other as well. Ancient/Barracks/Fountain sources remain Other. Unknown handles are
not guessed to be Towers. The first100 damage costs are hero-0.0029411764705882353,
creep-0.0058823529411764705, Tower-0.016666666666666666; creep/hero ratio is2 at equal
prior C. Unknown/environment100 retains the old Other cost-0.0008333333333333334.

## Potentials and terminal results

- Tower potential is `0.3*(mean own HP fraction - mean enemy HP fraction)`.
- Lane potential is `0.1*(mean own axis progress + mean enemy axis progress - 1)`.
- Potential DELTAS, unknown-cohort hold behavior and terminal lane closure are
  unchanged. With initial lane potential0, full-episode lane NET is0.
- The weak positive prewave center hint remains scale0.005; it is not replaced by
  a larger movement-delta hint.
- Fountain waiting/refund and progress debt/lease/latch coefficients/rules are
  unchanged. No purchase, mana, intent or additional anti-abuse conditions were added.
- Terminal rewards: Win+0.2, Loss-0.2, actual Draw0 (including native cap Draw),
  completed learner-task TimeCap-0.2. Labels remain distinct; Draw remains a nonwin
  in rolling mastery. Infrastructure/resource failures produce no game/reward.

## One-shot first-wave positioning

The public native wave period is900 ticks (bota037 config/rules.rs and wave_at).
Let P be public MatchInfo.pregame_ticks. The first-wave epoch is `[P,P+900)`.
During it, the first completed observation of an own living lane creep within
1500 inclusive of world center(9216,9216) triggers one assessment. Enemy/neutral/
dead creeps do not trigger it. If not yet assessed, the fallback is tick P+900.

The own live hero pays:

`-0.1 * clamp((Euclidean distance to center - 1500) / 1500, 0, 1)`.

Distance<=1500 costs0;2250 costs-0.05; distance>=3000 costs-0.1. Missing/dead body
costs the full-0.1. Radius boundaries use exact squared raw Fixed coordinates;
only the interpolation strictly between boundaries uses f64 sqrt.

Resolution is recorded even for zero cost. It never rearms after movement, retreat,
purchase, fountain reset, mana change, death or body-generation replacement. There
is no continuous or later-wave hero-position penalty and no cohort-ID tracking.

The first complete Snapshot/Events pair is baseline-free. A trigger already visible
on a first-wave baseline resolves without a retrospective/deferred charge. A
baseline at/after P+900 is also resolved. Earlier baselines stay pending until an
actual later trigger/deadline. Staging a Snapshot alone does not resolve the flag;
only its valid completed Events pair does. Clone and reward drains preserve it.
Finish expires any pending assessment without adding a late cost. Checked P+900
may lie beyond the native cap for an allowed artificial metadata baseline; it is
not clamped to an earlier tick and finish does not invent a second-wave assessment.

Telemetry adds `reward_tower_taken`, `tower_damage_taken`, `reward_opening_position`
and `opening_position_checks` (postbaseline assessments, including zero-cost ones).
Both new components participate in complete sums/finite validation; raw aggregation
is overflow-checked. None of these facts forces a policy action.

## Honest bounds without terminal dominance

Event budgets total positive0.14 and negative0.335. Prewave hint absolute bound0.005,
opening negative bound0.1, fountain negative bound0.093, stagnation negative bound0.2158.

For a NORMAL full native start with **initial tower potential0 AND lane potential0**:

- positive dense<=0.14+0.3+0.005 = **0.445**;
- negative dense magnitude<=0.335+0.3+0.005+0.1+0.093+0.2158 = **1.0488**;
- full lane net0; native zero initial potentials and the public wave period are
  verified in a small native fixture;
- The Win-vs-Draw gap is now **0.2**, and Win-vs-Loss **0.4**, both below the dense
  range1.4938. Even these normal starts have **no guaranteed winner-return dominance**.
  Tests construct adverse wins below favorable nonwins. No dense rescaling or clipping.

For GENERAL allowed late/primed observation baselines, tower delta may span0.6 and
terminal lane net may have magnitude0.1:

- positive dense<=**0.845**;
- negative/absolute dense<=**1.4488**, INCLUDING a possible negative0.005 prewave hint.

The generic APIs `MAP2_REWARD_DENSE_BOUND` and `MAP2_REWARD_POSITIVE_BOUND` expose
these general bounds. The narrower `MAP2_REWARD_NATIVE_NEGATIVE_BOUND` and
`MAP2_REWARD_NATIVE_POSITIVE_BOUND` are conditional on the stated zero potentials.
There is NO unconditional Win dominance for arbitrary primed baselines: a test
constructs a valid primed counterexample. The old v2 bound0.498 remains explicitly
historical, not a bound on the current reward profile.

## Appended features and semantic migration

Old globals0..89 retain their positions. Original nine remaining fractions still
occupy73..81; tower potential stays82, lane potential83, wait/progress85..89.
The tenth (Tower) remaining fraction is appended at90; opening-position-pending at91.
No IDs or extra mastery/controller bits enter the model. Unit84 is unchanged.

Current shape: global92, unit84, trunk2596x512,62 named tensors,**1,700,020 F32 parameters**.
Current identities:

| Contract | Version | Hash |
|---|---:|---:|
| Action | 5 | 10658390830565586343 |
| Feature | 20 | 9233114641639769206 |
| Model | 22 | 4891874295003631291 |
| PPO | 35 | 13569352384922890857 |
| League | 35 | 7630384836954837061 |
| Checkpoint | 10 | 2382613649322819763 |
| Reward | 6 | 1084583101075978392 |

Rules30, imitation20. Checkpoint mastery codec, thresholds, window rules and all
shapes are unchanged; new version/hash links bind the new reward meaning.
Old M19/M20/M21 runtimes/checkpoints are not silently accepted or relabelled.

The explicit pinned M19/u162 API `initialize_selected_m19_for_nonwin_reward` still
requires original runtime SHA9d0b88128bb4a74d636e0774ab53eeed2e2aea306c3ab0186a5698f41f92afea
and original frozen reward3 metadata/count1,698,996. It now inserts1024 positive-zero
scalars at old trunk global rows90..92 via Candle, preserving ALL original bits.
M16/M17 sources insert3584 scalars at85..92; M14 inserts10944 overall (unit and trunk).
Source metadata/descriptors and old artifacts remain frozen. Fresh model/optimizer/
mastery/RNG provenance is required; no gameplay/reward equivalence is claimed.
The explicit `initialize_selected_m21_for_terminal_reward` accepts only original
M21/u300 runtime SHA `b29752acf5c02687a54a63dce9d76480f8e97fc416ad9d4159c21a59e9109780`
and frozen reward5 metadata. All1,700,020 F32 parameters retain their bits, including
the actor and critic; no scaling or new rows. Adam/progress/mastery/RNG are fresh.
The new training run starts Weak with an empty window; u300 is ancestry, not resume
or inherited statistics. No M20 pin is added. See
[terminal02 training](../artifacts/temp/mastery-terminal02-20260914/REPORT.md).
