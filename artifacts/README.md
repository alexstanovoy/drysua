# Artifacts

Local, gitignored data:

- `weights/<name>/drysua.weights.safetensors`: runtime weights for play, eval and
  frozen opponents.
- `history/<campaign>/`: archived campaign logs, readable by `scripts/train.py report`.
- `temp/play-*/`: `play.sh` run directories (logs, replays, reward reports).
