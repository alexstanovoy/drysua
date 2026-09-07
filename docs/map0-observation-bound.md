# Map0 observation capacity repair

Superseded by [the 4096-cap / Feature10 repair](map0-feature10-initialization.md).
This document preserves the original 512-cap repair and its frozen-baseline ABI.

This repair changes observation capacity, not Teacher tactics, action selection,
simulator visibility, wave spawning, or model dimensions. The baseline patch is
`artifacts/temp/map0-observation-bound/frozen-baseline-observation-bound.patch`.

## Protocol research and chosen bounds

Audited simulator/protocol commit: `18db0f62d9a2b94e755c43fd29a959db204cc20b`.

- `bota-proto/src/codec.rs`: a four-byte little-endian length prefix and at most
  **4,194,304 payload bytes**. `FrameReader` rejects a larger declared payload from
  its header alone. `encode_frame` also rejects oversized output. The raw
  `decode_payload` helper does not enforce this framing limit itself.
- `bota-proto/src/view.rs`: `WorldView.units` and `.projectiles` are variable-length
  lists of **every visible** record, not selected model tokens. There is no protocol
  count constant of 256 units or 32 projectiles. Projectile records encode in 7–28
  postcard bytes; unit records vary with stats and nested abilities/items/effects.
  The payload cap limits a whole message, not each list independently.
- `bota-server/src/game/project.rs`: `World::view(team)` projects allied/visible
  entities without count truncation. `view_full()` is spectator-only. The shared
  entity allocator uses u32 indices, not a small gameplay population cap.
- `game/systems/wave.rs` and `game/config/wave.rs`: waves spawn in every lane and
  grow; there is no global unit/projectile count guard to justify the old limits.

| Contract | Old | New |
| --- | ---: | ---: |
| Accepted visible units / retained entity tracks | 256 | 512 |
| Accepted projectiles / continuous projectile histories | 32 | 512 |
| Model unit pointer rows | 96 | 96 |
| Model remembered-unit rows | 32 | 32 |
| Model projectile rows × fields | 32 × 20 | 32 × 20 |
| Observation rollback journal | 16 | 16 |
| Wire payload bytes | 4,194,304 | 4,194,304 |

512 is an explicit application safety limit, **not a claimed protocol maximum or
a guarantee for indefinitely long games**. It provides headroom above the reported
307-unit/39-projectile spectator peak and the reproduced seat peaks. A 513th unit
or projectile still fails atomically with respectively
`WorldView.units has 513 entries; limit is 512` or
`WorldView.projectiles has 513 entries; limit is 512`. Nested limits are unchanged.

Every accepted projectile is validated and remembered. The encoder selects the
first 32 complete tokens in the existing lexicographic encoded-semantic order;
identifiers are memory keys only. A previously unselected projectile retains age
and velocity if it later enters the selected rows. No observation is truncated.
Projectile history uses `Box<[Option<ProjectileObservation>; MAX_PROJECTILES]>`:
fixed allocation size, no growable history vector. An inline 512-entry history
multiplied across rollback states overflowed a release test thread's stack; the
boxed representation and an inline-size regression prevent that failure.

Worst-case projectile duplicate checks/history lookup remain bounded by 512²
comparisons, and selection by 512 × 32 token comparisons. A maximal projectile
list itself needs at most 14,336 record bytes plus framing/list overhead. The
tracker's existing entity eviction and all nested collection limits remain finite.
Teacher code scanning validated lists may rely on `MAX_PROJECTILES == 512` and
`MAX_TRACKED_ENTITIES == 512`; those are observation bounds, not model-row budgets.

## Exact legacy parity and ABI restrictions

For an identical accepted stream with at most 32 projectiles per observation and
**no eviction under the old 256-track memory bound**, tracker values, numeric
features, action candidates/masks, and fixed-shape model inputs are unchanged.
All numeric normalizers are unchanged, including visible-unit division by 256
with saturation. A SHA-256 golden regression captured before the fix covers every
feature scalar's f32 bits for 0/1/31/32 projectiles over two observations:
`f9aa155a880f099ae6b62650d3d8dc4549253ea5887469dc698ab99ba5cacedc`.

Bounding only the current view to 32 projectiles and 256 units is **not enough**
to promise whole-stream parity: more than 256 recent distinct unit handles can
trigger old memory eviction. The new tracker can retain those hidden tracks
longer. Streams previously rejected above either old capacity have no old output
to compare. No claim of unchanged old full-match outcomes is made.

| Schema | Version | Hash |
| --- | ---: | ---: |
| Action (unchanged) | 3 | `1755359086494840931` |
| Feature | 9 | `9669721049329356661` |
| Model | 9 | `832872366354465423` |
| PPO | 17 | `13743352113669513864` |
| League | 18 | `17865065897804281504` |
| Checkpoint envelope (unchanged) | 2 | `4581258024746721724` |

PPO/league rules audit remains 15. The cascade changes only schema identities,
not PPO objectives, rewards, optimizer settings, or league policy. Model parameter
count/order (1,684,724 F32 values) and all tensor shapes are unchanged; **semantic
checkpoint compatibility is not**. Existing strict loaders reject feature-v8 /
model-v8 / PPO-v16 weights and training resumes before mutation. No loader bypass,
metadata rewriting, full-neural-sample reinterpretation, or compatibility exception
was added. Teacher needs **no weights**. Any historical Teacher gate must bind the
exact known frozen source/binary and explicit Teacher policy; this patch adds no gate.

## Frozen replay and baseline handoff

Original freeze: `artifacts/temp/map0-neural-baseline-001`, drysua HEAD
`bd589a55be63427166ef561bf4b317b1dca8615a` plus its captured launcher working state.
Do not modify that freeze or substitute the concurrently changing Teacher.
Apply the patch to a **separate writable copy** of its `sources/drysua`, with the
frozen sibling `sources/bota`. Build offline/locked/release, then record new source
and binary hashes. Use explicit `play --policy teacher`, no weights. Neither root
launcher nor selected CLI default is part of this patch. A baseline rebuilt from
the mutable checkout is not this controlled opponent.

The ignored test
`frozen_map0_replay_accepts_every_seat_frame_including_first_33_projectiles` needs
`DRYSUA_MAP0_REPLAY` set to the original `matches/9470002/match.brp` (533,134,855
bytes; SHA-256 `da335fe5102bd25ff73a68e8271042d143231d1948968756373e24a73943519b`).
It replays recorded **orders** through unmodified Arena and supplies only Arena's
seat Snapshot/Events streams to trackers. Spectator snapshots/events are never
used as observations, and no hidden-state test hooks are used.

Old code fails at Dire tick **47,725**, 33 projectiles. Patched code accepts all
**48,626 snapshots per seat**, with first-33 ticks Radiant **48,626**, Dire **47,725**.
Seat unit/projectile peaks are Radiant **203/33**, Dire **211/39**. Synthetic Map0
tests additionally accept 307/39 and the exact 512/512 capacity. The original
post-Dire-disconnect interval remains in the replay; passing it is a capacity
regression result, not a completed fair self-play match or strength evaluation.

Evidence, red/green logs, and isolated source are under
`artifacts/temp/map0-observation-bound`. `verify.py` copies only the repair's owned
files onto the frozen source, excluding all concurrent Teacher/Tactical/CLI edits.
All checks use release builds, offline dependencies, and artifact-local target/tmp
directories. No simulator edits, dependencies, training runs, commits, or tags.
