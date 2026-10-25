"""Cache-only result replay and manifest: no models, network or test tuning."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import atomic_json, digest, now
from .dataset import KEYS


def verify(work: Path, config: dict):
    selected = json.loads((work / "selection.json").read_text())
    train = pd.read_parquet(work / "cache/train.parquet")
    test = pd.read_parquet(work / "cache/test.parquet")
    predicted = pd.read_csv(work / "predictions/final.csv")
    if len(predicted) != config["expected"]["test"] or predicted.duplicated(KEYS).any():
        raise ValueError("Final CSV coverage mismatch")
    if not predicted[KEYS].equals(test[KEYS]) or not predicted.label.isin([0, 1, 2]).all():
        raise ValueError("Prediction key order/label mismatch")
    if train.groupby("author").fold.nunique().max() != 1 or set(train.author) & set(test.author):
        raise ValueError("Author split leakage")
    replay = np.zeros((len(test), 3))
    for name, weight in selected["weights"].items():
        folder = work / "models" / name
        marker = json.loads((folder / "complete.json").read_text())
        for filename, sha in marker["outputs"].items():
            if digest(folder / filename) != sha:
                raise ValueError(f"Model artifact checksum mismatch: {name}/{filename}")
        with np.load(folder / "probabilities.npz", allow_pickle=False) as probs:
            replay += weight * probs["test"]
    labels = (replay * np.asarray(selected["class_scales"])).argmax(1)
    if not np.array_equal(labels, predicted.label.to_numpy()):
        raise ValueError("Cached probability replay differs from final CSV")
    outputs = [work / "predictions/final.csv", work / "selection.json", work / "report/training.json",
               work / "report/REPORT.md", work / "environment/manifest.json"]
    paths = [p for p in outputs if p.exists()]
    for parent in (work / "models", work / "environment", work / "report", work / "predictions"):
        paths += [p for p in parent.rglob("*") if p.is_file() and p not in paths and p.suffix != ".tmp"]
    manifest = "".join(f"{digest(p)}  {p.relative_to(work)}\n" for p in sorted(paths))
    (work / "SHA256SUMS").write_text(manifest)
    atomic_json(work / "report/verification.json", {"at": now(), "rows": len(predicted),
                "cache_replay": "passed", "author_split": "passed", "files_checksummed": len(paths),
                "dataset_not_packaged": True, "external_secret_not_packaged": True})
