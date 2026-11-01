from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--results", type=Path, default=Path("results/lab1"))
    ap.add_argument("--output", type=Path, default=Path("submission"))
    args = ap.parse_args()
    required = [args.results / "selected_blend.csv", args.results / "metrics.json",
                args.results / "selection.json", args.output / "Lab1_Rostislav_Manzyuk.pdf"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise SystemExit("Submission is incomplete: " + ", ".join(missing))
    metrics = json.loads((args.results / "metrics.json").read_text())
    if metrics.get("data_kind") != "official-release":
        raise SystemExit("Synthetic results cannot be packaged as an official submission")
    digest = hashlib.sha256((args.data / "labels.csv").read_bytes()).hexdigest()
    if digest != metrics["labels_sha256"]:
        raise SystemExit("Results refer to different release labels")
    truth = pd.read_csv(args.data / "labels.csv")
    truth = truth[truth.split == "test"]
    predictions = pd.read_csv(args.results / "selected_blend.csv")
    if len(truth) != len(predictions) or predictions.duplicated(["author", "post"]).any():
        raise SystemExit("Predictions do not cover every test post exactly once")
    merged = truth.merge(predictions, on=["author", "post"], validate="one_to_one",
                         suffixes=("", "_pred"))
    if len(merged) != len(truth) or not merged.label_pred.isin([0, 1, 2]).all():
        raise SystemExit("Prediction keys or classes do not match official test set")

    from kairos.metrics import score
    measured = score(merged.label, merged.label_pred)
    if any(measured[key] != metrics["test"]["selected_blend"][key] for key in measured):
        raise SystemExit("Prediction file disagrees with recorded metrics")
    args.output.mkdir(parents=True, exist_ok=True)
    for source in required[:3] + [args.results / "feature_importance.csv"]:
        shutil.copy2(source, args.output / source.name)
    print("Validated submission:", measured)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
