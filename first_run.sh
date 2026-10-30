#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -gt 2 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: $0 [path/to/lab1_core.7z.001] [data-directory]"
    echo "Extract non-media data, install dependencies in .venv, and run experiments."
    exit 0
fi

ARCHIVE="${1:-$ROOT_DIR/lab1_core.7z.001}"
DATA_DIR="${2:-$ROOT_DIR/data}"

if [[ ! -d "$DATA_DIR/posts" || ! -d "$DATA_DIR/authors" || ! -f "$DATA_DIR/labels.csv" || ! -f "$DATA_DIR/split.json" ]]; then
    if [[ ! -f "$ARCHIVE" ]]; then
        echo "Core archive not found: $ARCHIVE" >&2
        echo "Download all core volumes from https://kairos-labs.ai/dataset/ and pass the .001 path." >&2
        exit 1
    fi

    if command -v 7zz >/dev/null 2>&1; then
        SEVEN_ZIP="7zz"
    elif command -v 7z >/dev/null 2>&1; then
        SEVEN_ZIP="7z"
    else
        echo "Install 7-Zip (7zz or 7z) to extract the core archive." >&2
        exit 1
    fi

    mkdir -p -- "$DATA_DIR"
    # Keep all archive volumes together. 7-Zip prompts for the password.
    # Extract only the files consumed by the pipeline; leave media in the archive.
    "$SEVEN_ZIP" x "$ARCHIVE" "-o$DATA_DIR" -y -aoa \
        'authors/*' 'posts/*' labels.csv split.json
else
    echo "Using unpacked data in $DATA_DIR."
fi

if [[ ! -d "$DATA_DIR/posts" || ! -d "$DATA_DIR/authors" || ! -f "$DATA_DIR/labels.csv" || ! -f "$DATA_DIR/split.json" ]]; then
    echo "Expected authors/, posts/, labels.csv, and split.json in $DATA_DIR after extraction." >&2
    exit 1
fi

if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
    "${PYTHON:-python3}" -m venv "$ROOT_DIR/.venv"
fi

"$ROOT_DIR/.venv/bin/python" -m pip install -r "$ROOT_DIR/requirements.txt"
PYTHON="$ROOT_DIR/.venv/bin/python" exec bash "$ROOT_DIR/run_short.sh" "$DATA_DIR"
