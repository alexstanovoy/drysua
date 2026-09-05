# drysua v0.0.4

Compact learned Tactical v1 policy for Bota Map1. The canonical committed artifact
is `drysua.tactical.bin`, SHA-256:

```text
bfa19eb4ec2f11fd852f6f6fd6284be84aa0c0dc1b597a52fc00352d64ed5bd1
```

```sh
cargo build --release --locked --quiet --bin drysua --no-default-features
target/release/drysua play --policy tactical --weights-directory artifacts/v0.0.4
```

CPU live inference requires no CUDA, tensor-model initialization, or training.
Teacher remains available with `--policy teacher`, but is **not** the released
candidate policy. Explicit Tactical weights are mandatory; load failures never
fall back to Teacher. Use the tagged source: later Tactical schemas are not
implicitly compatible with these weights.

## Model and training

The 1,134-byte artifact is the exact Tactical v1 descriptor plus 172 little-endian
F32 parameters (`W1[8,16], b1[8], W2[4,8], b2[4]`). Its 16 seat-visible features
feed an eight-unit hard-tanh layer and four masked modes: Teacher, Fight, Recover,
Farm. Teacher retains economy, safety, legal decoding, persistence and rejection
handling. The runtime checks exact schema/length and finite values in [-4, 4].

The selected policy came from
`artifacts/temp/tactical-search-001/selection-017-1`. Round one used a bounded
16-member, 20-generation neural parameter search on rotating paired development
seeds; selection scored 24 wins / 6 losses / 2 timeouts over 32 games, and separate
confirmation scored 20 wins / 12 losses. Confirmation was not used to select the
weights. This release performed no additional training. `training.json` preserves
the original selected-policy report; `training-evidence.md` records provenance
and limitations. The artifact was imported and re-exported through validated
`TacticalPolicy` APIs and checked byte-for-byte, not schema-rewritten.

## Release gate

| Historical opponent | Wins | Losses | Win rate | Strict-majority gate |
|---|---:|---:|---:|---|
| v0.0.1 | 19 | 1 | 95% | Pass |
| v0.0.2 | 18 | 2 | 90% | Pass |
| v0.0.3 | 14 | 6 | 70% | Pass |

All 60 games used the registered ten seeds on both sides, actual separately
archived tagged bots, and the pinned TCP lockstep simulator commit
`18db0f62d9a2b94e755c43fd29a959db204cc20b`. Fixed limits: 30,000 ticks / 90 wall
seconds. No rejections, validation errors, draws, or timeouts occurred.

The registered requirement is strictly more than half of all scheduled games
against **each** opponent; it is not a pooled score. This release **does not meet
an 80% target against every opponent**: v0.0.3 was 70%, with 10/10 Radiant wins
and 4/10 Dire wins. Reused release/development seeds are not held-out evidence,
confidence guarantees, or a human-strength claim.

`gate.json` binds the pre-commit frozen-source gate, candidate and weight hashes.
The frozen source also reproduced all 32 archived selection games' outcomes and
both seats' order fingerprints. Full local evidence:
`artifacts/temp/tactical-v1-champion-crossplay-001/report.json` and
`artifacts/temp/tactical-v1-champion-001-provenance/`.

The annotated tag is created only after a clean committed-source build and a
second complete 60-game gate. That final evidence and the registry entry are
recorded in the follow-up registration commit, without changing the tagged source
or its canonical weights. V2 objective-policy experiments are excluded.

Clean release commit `7a2b758eee4098c646ce6f18e1d13e905b3212c3` was rebuilt from
`git archive` against the pinned simulator and passed the full 60-game schedule
before annotation. All 60 per-game outcomes and both client summaries exactly
matched the frozen-source run above. The artifact loaded successfully from the
committed archive. Final summary: `gate-clean.json`; full local evidence:
`artifacts/temp/v0.0.4-final-clean/report.json`. Release all-target/all-feature
Clippy passed with warnings denied; tests passed 473 with 8 ignored. Python gate
tests passed 37; formatting, dependency audit, and diff checks passed.
