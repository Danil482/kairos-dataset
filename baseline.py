"""Train the baseline on the train authors and score it on the test authors.

Gradient boosting over everything in the release except the words: what the post
is (length, hashtags, mentions, links, type, posting time), who wrote it (the
author's level and spread, how often they post, followers, profile) and when it
was crawled. Text enters only through summary features: formatting counts,
similarity to the author's earlier posts and relevance to their skills.

    python baseline.py
    python baseline.py --data D:/kairos --embedder tfidf

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
# is the physical count on most machines. Set before importing scikit-learn.
if sys.platform == "win32":
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) // 2)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.feature_selection import VarianceThreshold  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

from kairos import features  # noqa: E402
from kairos.data import (  # noqa: E402
    DATA_DIR,
    load_author_features,
    load_author_skills,
    load_followers,
    load_labels,
    load_media_features,
    load_posts,
)
from kairos.label import TYPICAL, author_level  # noqa: E402
from kairos.metrics import score  # noqa: E402
from kairos.semantic import build_semantic_features  # noqa: E402

SEED = 42
RESULTS = Path(__file__).resolve().parent / "results"

NUMERIC = (
    features.CONTENT_NUM
    + features.HISTORY_NUM
    + features.ARTEFACT_NUM
    + features.PROFILE_NUM
    + features.SEMANTIC_NUM
)
CATEGORICAL = features.CONTENT_CAT


def model() -> Pipeline:
    """Impute and encode the features, drop constant columns, then boost."""
    preprocess = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUMERIC),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("drop_constant", VarianceThreshold(threshold=0.0)),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    random_state=SEED,
                    class_weight="balanced",
                    max_iter=300,
                    early_stopping=True,
                ),
            ),
        ]
    )


def build_table(data_dir: Path | str, embedder: str) -> pd.DataFrame:
    """One row per labelled post: labels joined with all of its features."""
    labels = load_labels(data_dir)
    levelled = author_level(load_posts(data_dir))
    profiles = load_author_features(data_dir)
    media = load_media_features(data_dir)

    t = time.time()
    semantic = build_semantic_features(
        levelled, load_author_skills(data_dir), embedder, cache_dir=RESULTS / "cache"
    )
    print(f"\nsemantic features ({embedder}): {time.time() - t:.0f}s")
    print(semantic[features.SEMANTIC_NUM].notna().mean().round(3))

    # Sanity check that the optional feature sources actually loaded.
    print("\nProfile features:", profiles.shape)
    print(profiles.notna().sum())
    print("\nMedia features:", media.shape)
    print(
        media.head()
        if not media.empty
        else "No media records loaded. Check the manifest."
    )

    post_features = features.build(
        levelled,
        load_followers(data_dir),
        profile_features=profiles,
        media_features=media,
        semantic_features=semantic,
    )
    table = labels.merge(
        post_features, on=["author", "post"], how="left", validate="one_to_one"
    )
    if table["level"].isna().any():
        raise SystemExit("some labelled posts have no level: is the release complete?")
    return table


def score_all(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, np.ndarray]:
    """Score the model plus two model-free references; also return its predictions."""
    predicted = model().fit(train, train["label"]).predict(test)

    train_mix = np.bincount(train["label"], minlength=3) / len(train)
    random_guess = np.random.default_rng(SEED).choice(3, size=len(test), p=train_mix)

    results = {
        "baseline": score(test["label"], predicted),
        "always_typical": score(test["label"], np.full(len(test), TYPICAL)),
        "random_in_train_mix": score(test["label"], random_guess),
    }
    return results, predicted


def print_results(results: dict) -> None:
    print(f"\n{'':20s} {'macro-F1':>9s} {'QWK':>7s} {'accuracy':>9s}")
    for name, m in results.items():
        print(f"{name:20s} {m['macro_f1']:9.4f} {m['qwk']:7.4f} {m['accuracy']:9.4f}")


def write_outputs(
    test: pd.DataFrame, predicted: np.ndarray, n_train: int, results: dict
) -> None:
    (RESULTS / "predictions").mkdir(parents=True, exist_ok=True)
    test[["author", "post"]].assign(label=predicted).to_csv(
        RESULTS / "predictions" / "baseline.csv", index=False
    )
    summary = {
        "split": "test",
        "seed": SEED,
        "class_weight": "balanced",
        "n_train": n_train,
        "n_test": len(test),
        "scikit_learn": sklearn.__version__,
        "results": results,
    }
    (RESULTS / "baseline.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train and score the baseline.")
    ap.add_argument(
        "--data", default=DATA_DIR, help="unpacked release (default: data/)"
    )
    ap.add_argument(
        "--embedder",
        default="model2vec",
        choices=["model2vec", "sbert", "tfidf"],
        help="text embedding backend (default: model2vec, fastest)",
    )
    args = ap.parse_args()
    t0 = time.time()

    table = build_table(args.data, args.embedder)
    train, test = table[table["split"] == "train"], table[table["split"] == "test"]
    print(
        f"train {len(train):,} posts / {train['author'].nunique():,} authors   "
        f"test {len(test):,} posts / {test['author'].nunique():,} authors",
        flush=True,
    )

    results, predicted = score_all(train, test)
    print_results(results)
    write_outputs(test, predicted, len(train), results)
    print(
        f"\nwrote results/baseline.json and results/predictions/baseline.csv "
        f"({time.time() - t0:.0f}s)"
    )


if __name__ == "__main__":
    main()
