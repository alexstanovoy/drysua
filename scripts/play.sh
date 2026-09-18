#!/usr/bin/env bash
set -euo pipefail

if ! command -v python3 >/dev/null 2>&1; then
    printf 'play: Python 3 is required for process supervision.\n' >&2
    exit 1
fi

# Canonical tracked entry point; the workspace-root play.sh is a convenience shim
# that forwards to the same play_match.py.
scripts=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
exec python3 -B "$scripts/play_match.py" "$@"
