from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


def main():
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Reproduce the frozen Lab 1 experiment and submission")
    ap.add_argument("--data", type=Path, default=root / "data")
    ap.add_argument("--config", type=Path, default=root / "lab1_config.json")
    args = ap.parse_args()
    data = args.data.resolve()
    config = json.loads(args.config.read_text())

    def run(*command):
        subprocess.run([sys.executable, *map(str, command)], cwd=root, check=True)

    run("-m", "kairos.label", "--data", data)
    embedding_args = []
    if config.get("embeddings"):
        encoder = config["embeddings"]
        run("embed_lab1.py", "--data", data, "--revision", encoder["revision"],
            "--batch-size", encoder.get("batch_size", 24))
        embedding_args = ["--embeddings", ".cache/embeddings",
                          "--post-components", config["post_components"],
                          "--profile-components", config["profile_components"]]
    run("train_lab1.py", "--data", data, "--depth", config["depth"],
        "--iterations", config["iterations"], "--text-c", config["text_c"],
        "--threads", config.get("threads", 8), "--selection", "submission/selection.json", *embedding_args)
    run("evaluate.py", "--data", data, "--predictions", "results/lab1/selected_blend.csv")
    run("predict_lab1.py", "--data", data)
    original = pd.read_csv(root / "results/lab1/selected_blend.csv")
    repeated = pd.read_csv(root / "results/lab1/reproduced.csv")
    pd.testing.assert_frame_equal(original, repeated)
    run("make_report.py")
    run("tools/prepare_submission.py", "--data", data)
    print("Complete: submission/")


if __name__ == "__main__":
    main()
