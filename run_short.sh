#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -gt 1 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: $0 [data-directory]"
    echo "Run experiments using unpacked data and installed dependencies."
    exit 0
fi

DATA_DIR="${1:-$ROOT_DIR/data}"

if [[ ! -d "$DATA_DIR/posts" || ! -d "$DATA_DIR/authors" || ! -f "$DATA_DIR/labels.csv" || ! -f "$DATA_DIR/split.json" ]]; then
    echo "Missing dataset in $DATA_DIR. Run first_run.sh with the core archive first." >&2
    exit 1
fi

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

# baseline.py preprocesses features, trains the enabled models, and saves metrics
# and predictions for every enabled configuration.
exec "$PYTHON_BIN" "$ROOT_DIR/baseline.py" --data "$DATA_DIR"
