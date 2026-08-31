#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
instrumented="$root/target/pgo-instrumented"
profile_data="$root/target/pgo-data"
optimized="$root/target/pgo-optimized"
profdata="$root/target/drysua.profdata"
llvm_profdata="$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | awk '/host:/ { print $2 }')/bin/llvm-profdata"

test -x "$llvm_profdata" || {
    printf '%s\n' 'llvm-profdata is unavailable; run: rustup component add llvm-tools-preview' >&2
    exit 1
}

rm -rf "$instrumented" "$profile_data" "$optimized"
mkdir -p "$profile_data"

RUSTFLAGS="-Cprofile-generate=$profile_data" \
    CARGO_TARGET_DIR="$instrumented" \
    cargo build --manifest-path "$root/Cargo.toml" --release --all-features --bin drysua-profile

LLVM_PROFILE_FILE="$profile_data/drysua-%p-%m.profraw" \
    "$instrumented/release/drysua-profile" --arenas 2 --ticks 1000 --training-updates 1

"$llvm_profdata" merge -o "$profdata" "$profile_data"/*.profraw

RUSTFLAGS="-Cprofile-use=$profdata -Cllvm-args=-pgo-warn-missing-function" \
    CARGO_TARGET_DIR="$optimized" \
    cargo build --manifest-path "$root/Cargo.toml" --release --all-features --bin drysua

printf 'PGO binary: %s\n' "$optimized/release/drysua"
