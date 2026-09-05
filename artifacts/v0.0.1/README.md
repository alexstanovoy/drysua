# drysua v0.0.1 — playable Teacher baseline

This version uses the deterministic, seat-visible Shadow Fiend Teacher, not trained
neural weights. No GPU or weights are required. Map1 is the supported demo target.
The default CLI policy remains hybrid; explicitly select `--policy teacher`.

## Play

From the parent `bots` directory, start these in separate terminals:

```sh
cargo run --release --quiet --manifest-path bota/Cargo.toml -p bota-server -- --map 1 --players 2 --mode realtime --seed 9000001
```

```sh
cargo run --release --quiet --manifest-path drysua/Cargo.toml --bin drysua -- play --policy teacher --addr 127.0.0.1:4455 --name drysua
```

```sh
cargo run --release --quiet --manifest-path bota/Cargo.toml -p bota-client -- --addr 127.0.0.1:4455 --name human
```

In the human lobby select a hero with 1–5 (3 is Shadow Fiend), then press R to ready.
Join the human before the bot to reverse sides. Use compatible simulator commit
`18db0f62d9a2b94e755c43fd29a959db204cc20b`; protocol compatibility is not assumed
across arbitrary simulator revisions. The drysua source is the commit introducing
this document; this is a source release, not a prebuilt binary bundle.

## Evidence and limitations

- Map1 builtin evaluation: 6/6 wins against passive Weak, both sides at seeds
  9000001, 9000002, 9000003; zero rejected orders.
- Real TCP lockstep at seed 9000001: wins from both sides at ticks 7973 and 11492;
  521 and 995 orders respectively, zero rejections. These match builtin evaluation.
- Real-time 30 Hz TCP smoke: 1000 ticks, 33 decisions, 6 orders, zero rejections.
- Complete terminal replays were decoded, not visually reviewed.
- These checks establish functional gameplay, not competitive strength against humans.
  Map1 wins end at tower destruction; they are not full-map Ancient victories.
- No claim of learned strength: the experimental PPO weights failed quality evaluation
  and are deliberately excluded.

Local diagnostic logs and replays are in `artifacts/temp/teacher-live-*` and
`artifacts/temp/teacher-release-evaluation.*`; they are not required to play.
