"""Compare feature configurations and classifiers on the held-out authors.

Text configurations add post vectors or comparisons with earlier posts to the
baseline features; the text-only configuration uses SVD vectors alone. Media is
not used.

    python baseline.py
    python baseline.py --data D:/kairos

Writes one metrics CSV per classifier and one prediction CSV per model/configuration
pair, in the format evaluate.py reads.
"""

from __future__ import annotations

import argparse
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
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from scipy import sparse  # noqa: E402

from kairos import features  # noqa: E402
from kairos.data import DATA_DIR, load_followers, load_labels, load_posts  # noqa: E402
from kairos.label import TYPICAL, author_level  # noqa: E402
from kairos.metrics import score  # noqa: E402

SEED = 42
RESULTS = Path(__file__).resolve().parent / "results"
BASELINE = (
    features.CONTENT_NUM
    + features.HISTORY_NUM
    + features.ARTEFACT_NUM
    + features.CONTENT_CAT
)

SUCCESS = (
    BASELINE
    + features.TEXT_COMPARISON_TFIDF_NUM
    + features.TEXT_COMPARISON_E5_NUM
    + features.TEXT_COMPARISON_BM25_NUM
    + features.TEXT_E5_NUM
    + features.ESTIMATES_NUM
)

CONFIGURATIONS = {
    # "baseline": BASELINE,
    # "estimates": BASELINE + features.ESTIMATES_NUM,
    # "formatting": BASELINE + features.FORMATTING_NUM,
    # "text_bm25": BASELINE + features.TEXT_COMPARISON_BM25_NUM,
    # "text_tf_idf": BASELINE + features.TEXT_TFIDF_NUM,
    # "text_tf_idf_truncatedsvd": BASELINE + features.TEXT_SVD_NUM,
    # "text_only_tf_idf_truncatedsvd": features.TEXT_SVD_NUM,
    # "text_multilingual_e5_small": BASELINE + features.TEXT_E5_NUM,
    # "text_tfidfsvd_e5": BASELINE + features.TEXT_SVD_NUM + features.TEXT_E5_NUM,
    # "comparison_tfidf": BASELINE + features.TEXT_COMPARISON_TFIDF_NUM,
    # "comparison_e5": BASELINE + features.TEXT_COMPARISON_E5_NUM,
    # "comparison_bm25": BASELINE + features.TEXT_COMPARISON_BM25_NUM,
    # "comparison_tfidf_e5": BASELINE
    # + features.TEXT_COMPARISON_TFIDF_NUM
    # + features.TEXT_COMPARISON_E5_NUM,
    # "text_comparison": BASELINE
    # + features.TEXT_E5_NUM
    # + features.TEXT_COMPARISON_TFIDF_NUM
    # + features.TEXT_COMPARISON_E5_NUM
    # + features.TEXT_COMPARISON_BM25_NUM,
    # "text_comparison_estimates": SUCCESS,
    # "success_formatting": SUCCESS + features.FORMATTING_NUM,
    "except_text": BASELINE
    + features.TEXT_COMPARISON_TFIDF_NUM
    + features.TEXT_COMPARISON_BM25_NUM
    + features.TEXT_COMPARISON_E5_NUM
    + features.FORMATTING_NUM
    + features.ESTIMATES_NUM,
    "except_text_another_order": features.ESTIMATES_NUM
    + features.TEXT_COMPARISON_BM25_NUM
    + BASELINE
    + features.TEXT_COMPARISON_TFIDF_NUM
    + features.FORMATTING_NUM
    + features.TEXT_COMPARISON_E5_NUM,
}


def gradient_boosting() -> HistGradientBoostingClassifier:
    # Balanced weights counter the 20 / 60 / 20 class mix for macro-F1.
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        class_weight="balanced",
        random_state=SEED,
    )


def logistic_regression() -> LogisticRegression:
    return LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000)


MODELS = {
    "hist_gradient_boosting": gradient_boosting,
    # "logistic_regression": logistic_regression,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train and compare classifiers.")
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
    selected_features = list(
        dict.fromkeys(
            column for columns in CONFIGURATIONS.values() for column in columns
        )
    )
    train, test = features.prepare(
        train, test, selected_features, RESULTS, history=levelled
    )

    # Two answers that need no model, for scale.
    mix = np.bincount(train["label"], minlength=3) / len(train)
    reference = {
        "always_typical": score(test["label"], np.full(len(test), TYPICAL)),
        "random_in_train_mix": score(
            test["label"], np.random.default_rng(SEED).choice(3, size=len(test), p=mix)
        ),
    }

    (RESULTS / "predictions").mkdir(parents=True, exist_ok=True)
    (RESULTS / "metrics").mkdir(parents=True, exist_ok=True)
    results_by_model = {name: {} for name in MODELS}
    for name, feature_names in CONFIGURATIONS.items():
        transformer = features.preprocess(feature_names)
        x_train = transformer.fit_transform(train)
        x_test = transformer.transform(test)
        for model_name, make_model in MODELS.items():
            model_train, model_test = x_train, x_test
            if model_name == "hist_gradient_boosting" and sparse.issparse(x_train):
                model_train, model_test = x_train.toarray(), x_test.toarray()
            predicted = (
                make_model().fit(model_train, train["label"]).predict(model_test)
            )
            results_by_model[model_name][name] = score(test["label"], predicted)
            test[["author", "post"]].assign(label=predicted).to_csv(
                RESULTS / "predictions" / f"{model_name}__{name}.csv", index=False
            )
            print(f"{model_name} / {name}: done", flush=True)

    for model_name, results in results_by_model.items():
        results.update(reference)
        pd.DataFrame(
            [{"name": name, **metrics} for name, metrics in results.items()],
            columns=[
                "name",
                "macro_f1",
                "f1_below",
                "f1_typical",
                "f1_above",
                "qwk",
                "accuracy",
            ],
        ).to_csv(RESULTS / "metrics" / f"{model_name}.csv", index=False)

        name_width = max(20, *(len(name) for name in results))
        print(f"\n{model_name}")
        print(f"{'':{name_width}s} {'macro-F1':>9s} {'QWK':>7s} {'accuracy':>9s}")
        for name, m in results.items():
            print(
                f"{name:{name_width}s} {m['macro_f1']:9.4f} {m['qwk']:7.4f} {m['accuracy']:9.4f}"
            )

    print(
        f"\nwrote results/metrics/*.csv and results/predictions/*.csv "
        f"({time.time() - t0:.0f}s)"
    )


if __name__ == "__main__":
    main()
