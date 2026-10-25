"""Train the baseline on the train authors and score it on the test authors.

Gradient boosting over everything in the release except the words: what the post
is (length, hashtags, mentions, links, type, posting time), who wrote it (the
author's level and spread, how often they post, followers) and when it was
crawled. It never reads the text and never looks at the media.

    python baseline.py
    python baseline.py --data D:/kairos

Writes results/baseline.json and results/predictions/baseline.csv, in the format
evaluate.py reads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Gradient boosting asks joblib for the number of physical cores, joblib asks
# `wmic`, which Windows 11 no longer ships, and a warning and a traceback follow.
# joblib skips the question only for a cap below the logical count; half of it
# is the physical count on most machines.
if sys.platform == "win32":
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) // 2)))

import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from kairos import features  # noqa: E402
from kairos.data import DATA_DIR, load_followers, load_labels, load_posts  # noqa: E402
from kairos.label import TYPICAL, author_level  # noqa: E402
from kairos.metrics import score  # noqa: E402

SEED = 42
RESULTS = Path(__file__).resolve().parent / "results"
NUMERIC = features.CONTENT_NUM + features.HISTORY_NUM + features.ARTEFACT_NUM


def model() -> Pipeline:
    return Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), NUMERIC),
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore", min_frequency=20),
                            features.CONTENT_CAT,
                        ),
                    ]
                ),
            ),
            # Balanced class weights: the classes are 20 / 60 / 20, and an unweighted
            # model leans on "typical", which macro-F1 punishes.
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.08,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train and score the baseline.")
    ap.add_argument(
        "--data", default=DATA_DIR, help="unpacked release (default: data/)"
    )
    args = ap.parse_args()
    t0 = time.time()

    labels = load_labels(args.data)
    levelled = author_level(load_posts(args.data))
    table = labels.merge(
        features.build(levelled, load_followers(args.data)),
        on=["author", "post"],
        how="left",
        validate="one_to_one",
    )
    if table["level"].isna().any():
        raise SystemExit("some labelled posts have no level: is the release complete?")

    train, test = table[table["split"] == "train"], table[table["split"] == "test"]
    print(
        f"train {len(train):,} posts / {train['author'].nunique():,} authors   "
        f"test {len(test):,} posts / {test['author'].nunique():,} authors",
        flush=True,
    )

    predicted = model().fit(train, train["label"]).predict(test)
    results = {"baseline": score(test["label"], predicted)}
    # Two answers that need no model, for scale.
    mix = np.bincount(train["label"], minlength=3) / len(train)
    results["always_typical"] = score(test["label"], np.full(len(test), TYPICAL))
    results["random_in_train_mix"] = score(
        test["label"], np.random.default_rng(SEED).choice(3, size=len(test), p=mix)
    )

    print(f"\n{'':20s} {'macro-F1':>9s} {'QWK':>7s} {'accuracy':>9s}")
    for name, m in results.items():
        print(f"{name:20s} {m['macro_f1']:9.4f} {m['qwk']:7.4f} {m['accuracy']:9.4f}")

    (RESULTS / "predictions").mkdir(parents=True, exist_ok=True)
    test[["author", "post"]].assign(label=predicted).to_csv(
        RESULTS / "predictions" / "baseline.csv", index=False
    )
    (RESULTS / "baseline.json").write_text(
        json.dumps(
            {
                "split": "test",
                "seed": SEED,
                "class_weight": "balanced",
                "n_train": len(train),
                "n_test": len(test),
                "scikit_learn": sklearn.__version__,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"\nwrote results/baseline.json and results/predictions/baseline.csv "
        f"({time.time() - t0:.0f}s)"
    )


if __name__ == "__main__":
    main()
