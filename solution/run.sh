#!/usr/bin/env bash
# Generic Docker entrypoint; server credentials and weights stay outside the repo.
set -euo pipefail
command=${1:?Use predict, train, fetch, evaluate, or verify-weights}
shift
case "$command" in predict|train|fetch|evaluate|verify-weights) ;; *) exit 2;; esac
: "${WEIGHTS:?Set WEIGHTS to a local directory outside Git}"
[[ $(id -u) != 0 ]] || { printf 'Run as a non-root user.\n' >&2; exit 2; }
image=${KAIROS_IMAGE:-kairos-lab1}
work=${WORK:-"$PWD/reproduce"}
mkdir -p "$work" "$WEIGHTS"
work=$(cd "$work" && pwd -P)
weights=$(cd "$WEIGHTS" && pwd -P)
access=readonly
[[ "$command" != train && "$command" != fetch ]] || access=
args=(run --rm --user "$(id -u):$(id -g)" --cpus 8 --memory 32g --memory-swap 32g
      --pids-limit 512 --shm-size 2g --read-only --cap-drop ALL --security-opt no-new-privileges
      --tmpfs /tmp:rw,nosuid,nodev,size=2g --env HF_HOME=/work/huggingface --env XDG_CACHE_HOME=/work/xdg
      --mount "type=bind,src=$work,dst=/work"
      --mount "type=bind,src=$weights,dst=/weights${access:+,$access}")
if [[ "$command" == fetch ]]; then
    args+=(--network bridge)
else
    args+=(--network none)
fi
if [[ "$command" == train || "$command" == predict ]]; then
    : "${GPU_UUID:?Set GPU_UUID to the exact UUID of one allocated BF16 GPU}"
    [[ "$GPU_UUID" =~ ^GPU-[a-fA-F0-9-]+$ ]] || { printf 'Expected one GPU UUID.\n' >&2; exit 2; }
    args+=(--gpus "device=$GPU_UUID")
fi
if [[ "$command" == train || "$command" == predict || "$command" == evaluate ]]; then
    : "${DATA:?Set DATA to the unpacked official release}"
    data=$(cd "$DATA" && pwd -P)
    args+=(--mount "type=bind,src=$data,dst=/inputs,readonly")
fi
if [[ "$command" == evaluate ]]; then
    exec docker "${args[@]}" --entrypoint python "$image" evaluate.py --data /inputs --predictions /work/predictions.csv "$@"
fi
options=("$command" --weights /weights --work /work/cache)
[[ "$command" != predict && "$command" != train ]] || options+=(--data /inputs)
[[ "$command" != predict ]] || options+=(--output /work/predictions.csv)
exec docker "${args[@]}" "$image" "${options[@]}" "$@"
