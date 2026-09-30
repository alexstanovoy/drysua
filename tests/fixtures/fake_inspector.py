"""Native `checkpoint-inspect` stand-in reading the fake trainer's checkpoint."""
import hashlib
import json
from pathlib import Path
import sys

SCHEMA = "drysua-checkpoint-inspection/v1"


def main(arguments):
    if arguments == ["checkpoint-inspect", "--contract"]:
        print(json.dumps({"schema": SCHEMA, "kind": "contract", "model": {"hash": "0123456789abcdef"},
                          "capabilities": {"inspection": True, "annealed_history": True, "read_only": True,
                                           "controller_run_kind": "train-annealed"}}))
        return 0
    directory = Path(arguments[arguments.index("--checkpoint-directory") + 1])
    updates = json.loads((directory / "checkpoint.meta").read_text())["updates"]
    digest = hashlib.sha256(str(updates).encode()).hexdigest()
    print(json.dumps({
        "schema": SCHEMA, "kind": "train-annealed", "checkpoint": "fake",
        "identity": {"scope_sha256": "a" * 64, "manifest_sha256": digest, "tensor_sha256": digest,
                     "runtime_sha256": digest},
        "model": {"hash": "0123456789abcdef"}, "run": {"command_line": "train-annealed"}, "ppo": {"environments": 2},
        "progress": {"updates": updates, "optimizer_steps": updates, "rollout_samples": 100 * updates,
                     "games": 2 * updates},
        "adaptive": None, "files": [{"path": "checkpoint.meta", "size": 1, "sha256": digest}],
        "history": {"verified": True, "kind": "fixed", "snapshot_count": updates},
        "runtime_status": "matched", "runtime_matches_model": True, "recovery_required": False}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
