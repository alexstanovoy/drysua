# Public-seat laning diagnostic

`src/laning_evaluation.rs` is declared in `lib.rs` and its public types are
re-exported in the usual crate style. `src/tests/laning.rs` imports those public
types through `crate`, without a duplicate path-loaded module. The implementation
imports only the public protocol and the standard library, not controllers,
trainers, or simulator internals. The tests use the existing public `Arena`
adapter. No new production CLI, dependency, simulator change, or release gate is
required.

The initial process runner is research tooling in
`artifacts/temp/laning_isolated.rs` / `laning_diagnostic.rs`, rather than another
deployment framework. It connects a real candidate executable over framed TCP
to the public Arena adapter. Each proxy receives only its own fogged `WorldView`;
each observer consumes one seat's `Snapshot` / `Events` stream, never their union.
Candidate weights are copied byte-for-byte and loaded only by that executable.

## Cohort and opponents

- Map1, seeds **9240101 / 9240102**, candidate on both sides, three proxies: 12 games.
- Absolute tick cap **6300**, including the public 900-tick pregame.
- Proxy decisions at **1, 4, 7, ...**, with persistent orders and a 24-tick
  attack/cast commitment. Emergency recovery may interrupt that commitment, but
  never bypasses the three-tick cadence.
- `RightClick`: conservative last-hit/deny opportunities, otherwise right-click
  visible heroes outside their building protection.
- `RazeFarm`: the same pressure plus ready, affordable, facing-aligned razes at
  heroes or lane creeps. Uses public ability ranges and a conservative hit circle.
- `LanePush`: attacks lane creeps and advances towards the opposing tower.
- All learn available abilities and recover below 55% HP until at least 90% HP.
  No item build, opponent model, neural weights, or human-strength calibration.
- During pregame the proxies learn, move and recover, but do not initiate combat.
  A candidate can exploit that passivity; early wins are not human-strength proof.

The initial no-recovery scripts died twice before tick 3000. They exercised
combat but supplied no data for the requested phase. A failing survival test
preceded the recovery change. This was a diagnostic-quality fix, not an attempt
to tune proxy wins. Tests require real damage and casts, not wins.

## What the columns mean

**Direct, seat-visible measurements:**

- `hero_damage`, `hero_magical_damage`: positive `Damaged.amount` only when the
  source and target are opposing scoreboard hero identities. Retains generational
  identities through death; excludes NPC sources, creep targets, healing and
  friendly fire. Does not use `SlotStats.hero_damage` (zero in the pinned server).
- `hero_physical_hits`, `hero_magical_hits`: corresponding positive damage events,
  not orders issued or animations. `unit_physical_hits` separately includes creep
  damage/denies to verify that a push proxy actually attacks.
- `hero_damage_taken`: enemy-hero damage visible to this seat. Fog can make this
  differ from the other seat's outgoing total; do not add or reconcile the streams
  by inventing unseen events.
- `confirmed_casts`: own non-passive `AbilityCast` with a same-snapshot cooldown
  increase, once per ability per tick. A Learn event alone is not a cast, even
  while an earlier cooldown is running. This is a **lower bound**: a caster dying
  in that tick can lose its cooldown evidence. Events do not identify the ability
  responsible for damage, so this is not a cast-hit rate or proof of wasted mana.
- `last_hits`, `denies`, `deaths`: latest own public scoreboard. `winner`: public
  `MatchOver` only. `None` at the cap is **censored**, not a draw or a proxy win.
- `rejections`: server-reported rejected orders, not speculative legality labels.

**Defined geometric measurements / inferred behavior labels:**

- `first_center_tick`: first live hero sample within 1500 of the midpoint of the
  two initially public lane towers. This is a lane-arrival proxy, not the exact
  lane bend, first-wave contact or experience earned. No assumed 8192-unit map.
- Phase is **3000..=6000**, inclusive. `phase_ticks` is the observation denominator;
  `phase_alive_ticks` counts living own-hero samples; `phase_xp_ticks` additionally
  requires a visible living enemy lane creep within **1500**, inclusive. Report
  both XP/alive and XP/observed fractions; the former alone hides deaths.
- `healthy_behind_tower_ticks`: phase samples behind the initial own tower by
  more than 300 in signed `x+y` progress, HP at least 60%, and no seat-visible
  damage taken in the last 90 ticks. This is a **posture proxy**, not proof of a
  retreat order, lack of danger, or an irrational decision. Fog conceals threats.
- Creep aggro targets are absent from this protocol. Hero hits measure exercised
  pressure; there is **no direct aggro-success metric** or inferred hidden target.
- An unobserved phase has denominator zero and must be treated as **N/A**, not 0%
  occupancy. Reports are complete only after a successful runner exit and 24 rows.

## Reproduce with the archived simulator dependencies

Run from `drysua/`. These commands avoid mutable controller/schema code and build
no debug dependencies. The existing clean release archive supplies the pinned
simulator (`18db0f62d9a2b94e755c43fd29a959db204cc20b`). Do not pick arbitrary stale
rlibs from the mutable target directory: an early diagnostic attempt did so and
the archived bot correctly failed on an incompatible ItemView schema.

```sh
SIM=artifacts/temp/v0.0.4-final-clean/target-server/release/deps
rustc --edition=2024 -O artifacts/temp/laning_isolated.rs \
  -L dependency="$SIM" -L dependency=target/release/deps \
  --extern bota_proto="$SIM/libbota_proto-3c1e12949d7112ff.rlib" \
  --extern bota_server="$SIM/libbota_server-fde637c0e5853959.rlib" \
  --extern clap=target/release/deps/libclap-ea9d3b69c71de8d3.rlib \
  --extern sha2=target/release/deps/libsha2-4b86a10e89f5b12f.rlib \
  -o artifacts/temp/laning_isolated

rustc --edition=2024 -O --test artifacts/temp/laning_isolated.rs \
  -L dependency="$SIM" \
  --extern bota_proto="$SIM/libbota_proto-3c1e12949d7112ff.rlib" \
  --extern bota_server="$SIM/libbota_server-fde637c0e5853959.rlib" \
  -o artifacts/temp/laning_isolated_tests
artifacts/temp/laning_isolated_tests --quiet
```

Once the shared crate builds, the normal test command is
`cargo test --release --features builtin --lib tests::laning --quiet`.

## Before / after commands

Output directories must be new. Each contains an immutable candidate/weights
snapshot, their SHA256s, the harness executable/hash, per-game process logs, and
`report.tsv`. Keep the same harness executable for both halves.

```sh
artifacts/temp/laning_isolated \
  --binary artifacts/temp/v0.0.4-clean-build/target-cpu/release/drysua \
  --policy tactical --weights artifacts/v0.0.4 \
  --output artifacts/temp/laning-before

# After the controller/schema owners finish integration:
cargo build --release --locked --offline --bin drysua --quiet &&
artifacts/temp/laning-before/harness \
  --binary target/release/drysua \
  --policy tactical --weights PATH_TO_UPDATED_ACTUAL_TACTICAL_ARTIFACT \
  --output artifacts/temp/laning-after &&
diff -u artifacts/temp/laning-before/report.tsv artifacts/temp/laning-after/report.tsv
```

For an explicitly Teacher-only diagnostic, replace the policy/weights arguments
with `--policy teacher`; label it Teacher, not an updated tactical artifact.
Never migrate v0.0.4 weights into a changed schema and call that the old baseline.

### Fresh, untrained default Tactical v2 comparison

`artifacts/temp/laning_default_v2.rs` uses the newly built library's
`TacticalPolicy::default()` and `to_bytes()` APIs. Compile-time guards require
24 inputs, 8 hidden units, 4 modes, and 236 parameters. It checks the v2 descriptor,
serialization round trip, and constant Teacher scores before creating a new
artifact directory. It never opens the old v1 weights. `factory.txt` labels the
result **UNTRAINED default Tactical v2; Teacher selector** and records its hash.

After schema and economy integration, run from `drysua/` (new output directories):

```sh
cargo build --release --locked --offline --features builtin --lib --bin drysua --quiet &&
rustc --edition=2024 -O artifacts/temp/laning_default_v2.rs \
  -L dependency=target/release/deps \
  --extern drysua=target/release/libdrysua.rlib \
  --extern clap=target/release/deps/libclap-ea9d3b69c71de8d3.rlib \
  --extern sha2=target/release/deps/libsha2-4b86a10e89f5b12f.rlib \
  -o artifacts/temp/laning_default_v2 &&
artifacts/temp/laning_default_v2 \
  --output artifacts/temp/laning-default-v2-weights-001 &&
artifacts/temp/laning-v004-before-004/harness \
  --binary target/release/drysua --policy tactical \
  --weights artifacts/temp/laning-default-v2-weights-001 \
  --output artifacts/temp/laning-default-v2-after-001 &&
diff -u artifacts/temp/laning-v004-before-004/report.tsv \
  artifacts/temp/laning-default-v2-after-001/report.tsv
```

The retained executable fixes proxy behavior, metric definitions, seeds and
cadence across both runs. Its aggregate TSV cannot recover a true **raze miss
ratio**: casts are not separated by ability or target intent, and farming casts,
Requiem and multi-hit effects confound the aggregate. Report magical hero damage
events / confirmed casts as an explicitly labeled pressure-yield ratio instead;
do not call its complement a measured miss rate.

## Initial evidence and integration boundary

The archived v0.0.4 binary SHA256 is
`43a69a48ffad67aeebaeadcd3fb1d2f877ef34f0ce54d82f94f9cd16d7a58f9a`;
the unchanged weights SHA256 is
`bfa19eb4ec2f11fd852f6f6fd6284be84aa0c0dc1b597a52fc00352d64ed5bd1`.
The archived simulator rlib SHA256 is
`e69893a3745789857b4bf23cd40cb182f39cc22eb448b636e54334ac5a404a77`.

`artifacts/temp/laning-v004-before-004/report.tsv` is a completed real-process
baseline: all 12 games reach 6300, all seats have 3001 phase observations, and
there are no rejected orders. Candidate arrival is 1019..1031 versus proxy
122..127. Its rows exactly reproduce the previous `before-003` run. These are
not human-strength or release-approval results.

Candidate totals (four games per row; raw denominators, not averaged percentages):

| Proxy | Hero damage | Magical hero damage | Physical hero hits | Confirmed casts | XP / alive phase ticks | Healthy behind-tower ticks | LH | Deaths |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RightClick | 4106 | 2462 | 31 | 70 | 7905 / 12004 | 762 | 62 | 0 |
| RazeFarm | 4540 | 2006 | 49 | 75 | 8027 / 12004 | 854 | 70 | 0 |
| LanePush | 4485 | 3355 | 22 | 69 | 7721 / 12004 | 693 | 55 | 0 |

The initial shared build was blocked by the action-schema migration. After the
other owners integrated it, the ten normal Cargo laning tests and the default
Tactical Teacher-selection test passed, and the release library/production binary
built successfully. The public-module tests are no longer limited to the isolated
wrapper. Final release all-target/all-feature Clippy passed, as did the ten
filtered Cargo laning tests, the default-selector test, and four comparison-tool
regressions. Targeted formatting, `git diff --check`, and dependency audit passed.
No full match/release gate was run.

## Completed after: UNTRAINED default Tactical v2

The real updated production process completed the 12-game cohort using the exact
saved before-run harness executable, not a recompiled or retuned proxy. The
production source at build time delegates to the economy helper for the Wraith
Band/Tango opening and sustain, and includes pregame and 24x8x4 combat changes.
The factory calls `TacticalPolicy::default()` from that newly built library:
**236 parameters, 2282 bytes, no training and no v1 conversion**. This is a
combined-runtime/default-initialization comparison, not an ablation or evidence of
neural learning.

- After report: `artifacts/temp/laning-default-v2-after-001/report.tsv`.
- Factory provenance: `artifacts/temp/laning-default-v2-weights-001/factory.txt`.
- Detailed paired tables: `artifacts/temp/laning-default-v2-after-001/comparison.md`.
- After binary: `bb66246478e5b7d7d54cbe1fba1b6f2f85c7ab18e2589d9b6f44e827b562f7f9`.
- Fresh default weights: `65fadc1772be3009147c4052940ac7854bd41dd1dcdb80ef0a42784795dd3453`.
- Shared harness: `d1fb95684c800f308d4b69117191d8e7c8278faf4533a703d24924131e083d5b`.

**Arrival:** all twelve candidate arrivals improve from 1019..1031 to **122..134**
absolute ticks, before the 900-tick horn. All seats report zero rejections.
Both Radiant-vs-RazeFarm after games end at **2024** with a Radiant victory; the
other ten games reach **6300**. The two early terminals have **no phase samples**.

For fair phase and full-duration comparisons, the following table uses only the
**same ten pairs** that reach 6300 in both halves, excluding the corresponding
two before games as well. XP includes only living heroes near visible enemy lane
creeps; the observed-tick denominator prevents death time disappearing.

| Metric, ten matched full games | v0.0.4 | UNTRAINED default v2 |
|---|---:|---:|
| XP occupancy / observed phase | 20120/30010 = 67.04% | 20273/30010 = 67.55% |
| XP occupancy / alive phase | 20120/30010 = 67.04% | 20273/29530 = 68.65% |
| Healthy behind-tower posture / observed phase | 1954/30010 = 6.51% | 1105/30010 = 3.68% |
| Enemy hero damage dealt | 10345 | 22009 |
| Magical enemy hero damage | 6579 | 11609 |
| Enemy hero damage taken | 3391 | 7325 |
| Physical enemy hero hits | 75 | 213 |
| Confirmed casts | 174 | 235 |
| Magical hero damage events / confirmed casts | 80/174 = 45.98% | 144/235 = 61.28% |
| Last hits | 145 | 129 |
| Denies | 16 | 17 |
| Deaths | 0 | 1 |

### Remaining human-feedback weaknesses

1. **Arrival is repaired in this proxy definition, but lane occupancy barely
   improves overall:** +0.51 percentage points on the common observed phase.
   Do not substitute the larger alive-only percentage for that result.
2. **Radiant against LanePush remains a clear position/XP concern.** Both games
   reach 6300 with no candidate death, but XP occupancy falls from
   **4062/6002 (67.68%) to 3121/6002 (52.00%)**. Aggregate side-averaging hides this:
   Dire improves from 60.96% to 76.94% against the same proxy. This flags a wave/
   position investigation, not proven XP loss: faster wave clearing, displacement
   and vision also change this proxy, and XP earned is not retained in the TSV.
3. **More pressure is not an unqualified farming/safety improvement.** Last hits
   fall **145 to 129** on equal-duration pairs, hero damage taken more than doubles,
   and Dire dies once against RightClick at seed 9240101 (none in the before half).
   Fewer healthy behind-tower samples do not establish that every retreat choice
   is now safe or appropriate.
4. **Spell pressure improves, not proven raze accuracy.** The aggregate magical
   event/cast yield rises, but the true raze miss fraction remains **N/A** for this
   retained report format. Radiant-LanePush yield actually drops from 23/30 to
   31/50 despite greater damage. Farming/non-raze casts and missing cast evidence
   prevent calling this either a miss rate or a precise harass success rate.
5. **Aggro manipulation and human strength remain uncalibrated.** Physical hero
   hits demonstrate exercised attacks, not creep-target control. The early wins
   are versus simple, unequipped scripts, including pregame passivity, not humans.

Recompute (or verify and reprint) the paired tables with:

```sh
python3 artifacts/temp/laning_compare.py \
  artifacts/temp/laning-v004-before-004 \
  artifacts/temp/laning-default-v2-after-001 \
  artifacts/temp/laning-default-v2-weights-001
```

The helper validates the 24-row cohort, shared harness hash and default-export
hash. It creates `comparison.md` or verifies an identical existing result; it
never overwrites different evidence.
