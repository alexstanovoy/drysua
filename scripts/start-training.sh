#!/usr/bin/env bash
set -euo pipefail
umask 077

if (( $# < 1 || $# > 3 )); then
    printf 'usage: %s RUN_DIRECTORY [TOTAL_UPDATES] [fresh|resume|migrate]\n' "$0" >&2
    exit 2
fi

repository=$(realpath "$(dirname "$0")/..")
simulator_repository=$(realpath "$repository/../bota")
run_directory=$(realpath -m "$1")
total_updates=${2:-50000}
mode=${3:-fresh}
checkpoint_directory="$run_directory/checkpoint"
log_file="$run_directory/training.log"
pid_file="$run_directory/training.pid"
binary="$repository/target/release/drysua"

if [[ $run_directory == "$repository" || $run_directory == "$repository/"* \
    || $run_directory == "$simulator_repository" || $run_directory == "$simulator_repository/"* ]]; then
    printf 'RUN_DIRECTORY must be outside the source repository.\n' >&2
    exit 2
fi

maximum_updates=$((1000000000 / (4 * 8 * 132)))
if [[ ! $total_updates =~ ^[0-9]+$ ]] || (( total_updates < 1 || total_updates > maximum_updates )); then
    printf 'TOTAL_UPDATES must be in 1..%s for the fixed checkpoint-safe rollout.\n' "$maximum_updates" >&2
    exit 2
fi
if [[ $mode != fresh && $mode != resume && $mode != migrate ]]; then
    printf 'mode must be fresh, resume, or migrate.\n' >&2
    exit 2
fi
if [[ $mode == fresh && -e $run_directory ]]; then
    printf 'fresh run directory already exists: %s\n' "$run_directory" >&2
    exit 2
fi
if [[ $mode != fresh && ! -d $checkpoint_directory ]]; then
    printf 'resume checkpoint directory does not exist: %s\n' "$checkpoint_directory" >&2
    exit 2
fi
if [[ $mode != fresh ]]; then
    if [[ -L $run_directory || -L $checkpoint_directory || -L $log_file || -L $pid_file ]]; then
        printf 'resume paths must not be symbolic links.\n' >&2
        exit 2
    fi
    for directory in "$run_directory" "$checkpoint_directory"; do
        if [[ $(stat -c '%u' "$directory") != $EUID || $(stat -c '%a' "$directory") != 700 ]]; then
            printf 'resume directories must be owned by this user with mode 700: %s\n' "$directory" >&2
            exit 2
        fi
    done
    if [[ -e $log_file && ! -f $log_file ]] || [[ -e $pid_file && ! -f $pid_file ]]; then
        printf 'resume log and PID paths must be regular files when present.\n' >&2
        exit 2
    fi
    for file in "$log_file" "$pid_file"; do
        if [[ -e $file ]] && { [[ $(stat -c '%u' "$file") != $EUID ]] || [[ $(stat -c '%a' "$file") != 600 ]]; }; then
            printf 'resume files must be owned by this user with mode 600: %s\n' "$file" >&2
            exit 2
        fi
    done
fi
if [[ -f $pid_file ]]; then
    read -r previous_pid < "$pid_file"
    if [[ $previous_pid =~ ^[0-9]+$ ]] && kill -0 "$previous_pid" 2>/dev/null; then
        printf 'training process %s is already running.\n' "$previous_pid" >&2
        exit 2
    fi
fi

runtime_directory=${XDG_RUNTIME_DIR:-/run/user/$EUID}
if [[ ! -d $runtime_directory || -L $runtime_directory \
    || $(stat -c '%u' "$runtime_directory") != $EUID ]]; then
    printf 'a private user runtime directory is required for the CUDA device lock.\n' >&2
    exit 1
fi
device_lock="$runtime_directory/drysua-cuda-device-0.lock"
exec 9> "$device_lock"
if ! flock --nonblock 9; then
    printf 'CUDA device 0 already has a drysua training launcher or process.\n' >&2
    exit 1
fi

require_clean_tree() {
    local worktree=$1
    local name=$2
    if [[ -n $(git -C "$worktree" status --porcelain --untracked-files=normal) ]]; then
        printf '%s working tree must be clean before a provenance-bound build.\n' "$name" >&2
        exit 1
    fi
}

require_clean_tree "$repository" drysua
require_clean_tree "$simulator_repository" bota
if (( $(nproc) < 4 )); then
    printf 'training launcher requires at least four online CPU cores.\n' >&2
    exit 1
fi

available_kib=0
while read -r key value _; do
    if [[ $key == MemAvailable: ]]; then
        available_kib=$value
        break
    fi
done < /proc/meminfo
if (( available_kib < 16 * 1024 * 1024 )); then
    printf 'at least 16 GiB available RAM is required; found %s KiB.\n' "$available_kib" >&2
    exit 1
fi

gpu_free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=0)
gpu_free_mib=${gpu_free_mib//[[:space:]]/}
if [[ ! $gpu_free_mib =~ ^[0-9]+$ ]] || (( gpu_free_mib < 8192 )); then
    printf 'at least 8192 MiB free VRAM is required on CUDA device 0.\n' >&2
    exit 1
fi
gpu_compute_capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader --id=0)
gpu_compute_capability=${gpu_compute_capability//[[:space:]]/}
if [[ $gpu_compute_capability != 12.0 ]]; then
    printf 'CUDA device 0 must have compute capability 12.0; found %s.\n' "$gpu_compute_capability" >&2
    exit 1
fi

drysua_commit=$(git -C "$repository" rev-parse HEAD)
simulator_commit=$(git -C "$simulator_repository" rev-parse HEAD)
cuda_compiler=$(readlink -f "$(command -v nvcc)")
cuda_root=$(realpath "$(dirname "$cuda_compiler")/..")
if [[ ! -f $cuda_root/version.json ]]; then
    printf 'CUDA toolkit version manifest is missing: %s\n' "$cuda_root/version.json" >&2
    exit 1
fi
build_identity=$(
    {
        RUSTUP_TOOLCHAIN=1.98.0 rustc -vV
        RUSTUP_TOOLCHAIN=1.98.0 cargo -V
        "$cuda_compiler" --version
        sha256sum "$cuda_root/version.json"
        nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader --id=0
        printf 'CUDA_COMPUTE_CAP=120\nRUSTFLAGS=<empty>\n'
    } | sha256sum
)
build_fingerprint=${build_identity%% *}
drysua_provenance="$drysua_commit+build.$build_fingerprint"
printf 'Building bounded CUDA trainer with two compilation jobs.\n'
(
    cd "$repository"
    env -i \
        HOME="$HOME" \
        PATH="$PATH" \
        CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}" \
        RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}" \
        RUSTUP_TOOLCHAIN=1.98.0 \
        DRYSUA_GIT_COMMIT="$drysua_provenance" \
        BOTA_GIT_COMMIT="$simulator_commit" \
        CARGO_BUILD_JOBS=2 \
        CARGO_TARGET_DIR="$repository/target" \
        CUDA_COMPUTE_CAP=120 \
        CUDA_ROOT="$cuda_root" \
        CUDA_HOME="$cuda_root" \
        CUDA_PATH="$cuda_root" \
        RUSTFLAGS= \
        CARGO_ENCODED_RUSTFLAGS= \
        cargo build --quiet --locked --release --features builtin,cuda --bin drysua
)
if [[ $(git -C "$repository" rev-parse HEAD) != $drysua_commit \
    || $(git -C "$simulator_repository" rev-parse HEAD) != $simulator_commit ]]; then
    printf 'source commit changed during the provenance-bound build.\n' >&2
    exit 1
fi
require_clean_tree "$repository" drysua
require_clean_tree "$simulator_repository" bota

if [[ $mode == fresh ]]; then
    mkdir "$run_directory"
    mkdir "$checkpoint_directory"
    resume_argument=()
else
    if [[ $mode == migrate ]]; then
        resume_argument=(--resume --migrate-provenance)
    else
        resume_argument=(--resume)
    fi
fi

command=(
    "$binary" train-full
    --updates "$total_updates"
    --environments 4
    --rollout 8
    --epochs 1
    --minibatch 32
    --checkpoint-seconds 300
    --checkpoint-directory "$checkpoint_directory"
    --seed 9001
    --map 1
    --device cuda
    --device-ordinal 0
    "${resume_argument[@]}"
)

if [[ $mode == fresh ]]; then
    nohup taskset -c 0-3 nice -n 15 env OMP_NUM_THREADS=2 "${command[@]}" \
        > "$log_file" 2>&1 < /dev/null &
else
    nohup taskset -c 0-3 nice -n 15 env OMP_NUM_THREADS=2 "${command[@]}" \
        >> "$log_file" 2>&1 < /dev/null &
fi
training_pid=$!
pid_temporary=$(mktemp "$run_directory/.training.pid.XXXXXX")
printf '%s\n' "$training_pid" > "$pid_temporary"
mv -fT "$pid_temporary" "$pid_file"
sleep 2
if ! kill -0 "$training_pid" 2>/dev/null; then
    printf 'training exited during startup; inspect %s\n' "$log_file" >&2
    exit 1
fi

printf 'training PID: %s\nlog: %s\ncheckpoint: %s\n' \
    "$training_pid" "$log_file" "$checkpoint_directory"
