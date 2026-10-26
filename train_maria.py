"""Author-separated experiments and a frozen final evaluation for Lab 1.

CV uses only official train authors. Every outer fold has an inner author
calibration split for fusion weights and class decision multipliers.
The test labels are used for scoring only after the final model is frozen.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "6")
os.environ.setdefault("OMP_NUM_THREADS", "6")

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline

from kairos.data import DATA_DIR, load_followers, load_labels, load_posts, load_split
from kairos.metrics import score
from student_features import CATEGORY, SNAPSHOT, build_features

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "maria"
SEED = 20261025


def metrics(y, predicted):
    return {**score(y, predicted),
            "macro_f1": round(float(f1_score(y, predicted, average="macro", labels=[0, 1, 2], zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1, 2]).tolist(),
            "by_class": classification_report(y, predicted, labels=[0, 1, 2],
                target_names=["below", "typical", "above"], output_dict=True, zero_division=0)}


def text_model():
    return Pipeline([
        ("tfidf", FeatureUnion([
            ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=.98,
                max_features=45000, sublinear_tf=True, dtype=np.float32)),
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                max_features=45000, sublinear_tf=True, dtype=np.float32)),
        ])),
        ("classifier", SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=1000,
            tol=1e-4, average=True, class_weight="balanced", random_state=SEED)),
    ])


def boost(iterations):
    return CatBoostClassifier(iterations=iterations, depth=6, learning_rate=.06,
        loss_function="MultiClass", auto_class_weights="Balanced", l2_leaf_reg=5,
        random_seed=SEED, thread_count=6, verbose=False, allow_writing_files=False)


def text_input(table):
    # Media-only posts remain in the population and still get a prediction.
    return table.text.mask(table.text.str.strip().eq(""), "empty_post")


def calibrate(y, clean, text):
    best = None
    for text_weight in (0., .1, .25, .5):
        blended = (1 - text_weight) * clean + text_weight * text
        for below in (.8, 1., 1.2):
            for above in (.8, 1., 1.2):
                multipliers = [below, 1., above]
                value = f1_score(y, np.argmax(blended * multipliers, axis=1), average="macro", labels=[0, 1, 2])
                if best is None or value > best["calibration_macro_f1"]:
                    best = dict(text_weight=text_weight, multipliers=multipliers,
                                calibration_macro_f1=float(value))
    return best


def combine(clean, text, config):
    weight = config["text_weight"]
    return ((1 - weight) * clean + weight * text) * config["multipliers"]


def fit_predict(train, validation, numeric, iterations, snapshot_ablation=True):
    fitting, calibrating = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(train, groups=train.author))
    fitting, calibrating = train.iloc[fitting], train.iloc[calibrating]
    assert not set(fitting.author) & set(calibrating.author)
    assert not set(train.author) & set(validation.author)
    columns = numeric + CATEGORY
    history = boost(iterations).fit(fitting[columns], fitting.label, cat_features=CATEGORY)
    text = text_model().fit(text_input(fitting), fitting.label)
    config = calibrate(calibrating.label, history.predict_proba(calibrating[columns]),
                       text.predict_proba(text_input(calibrating)))
    history_prob = history.predict_proba(validation[columns])
    text_prob = text.predict_proba(text_input(validation))
    predictions = {
        "history": history_prob.argmax(axis=1),
        "tfidf": text_prob.argmax(axis=1),
        "fusion": combine(history_prob, text_prob, config).argmax(axis=1),
    }
    if snapshot_ablation:
        snapshot = boost(iterations).fit(fitting[columns + SNAPSHOT], fitting.label, cat_features=CATEGORY)
        predictions["history_plus_snapshot"] = snapshot.predict_proba(validation[columns + SNAPSHOT]).argmax(axis=1)
    bundle = dict(history=history, text=text, config=config, numeric=numeric,
                  fitting_authors=sorted(fitting.author.unique()),
                  calibration_authors=sorted(calibrating.author.unique()),
                  n_fitting=len(fitting), n_calibration=len(calibrating))
    return predictions, bundle


def bootstrap_authors(table, column, reference=None, repeats=500):
    # Resample entire authors, preserving dependence among their posts.
    authors = sorted(table.author.unique())
    matrices = []
    references = []
    for _, group in table.groupby("author", sort=True):
        matrices.append(confusion_matrix(group.label, group[column], labels=[0, 1, 2]))
        if reference:
            references.append(confusion_matrix(group.label, group[reference], labels=[0, 1, 2]))
    matrices = np.asarray(matrices)
    references = np.asarray(references)
    rng = np.random.default_rng(SEED)
    values = []

    def macro(matrix):
        denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
        return float(np.divide(2 * matrix.diagonal(), denominator,
            out=np.zeros(3, dtype=float), where=denominator != 0).mean())

    for _ in range(repeats):
        chosen = rng.integers(0, len(authors), len(authors))
        value = macro(matrices[chosen].sum(axis=0))
        if reference:
            value -= macro(references[chosen].sum(axis=0))
        values.append(value)
    return dict(low=float(np.quantile(values, .025)), high=float(np.quantile(values, .975)),
                resamples=repeats, unit="author")


def load_table(data):
    labels = load_labels(data)
    split = load_split(data)
    cutoffs = split["cutoffs_on_log_residual"]
    print("Building strictly past-only features...", flush=True)
    table, numeric = build_features(load_posts(data), load_followers(data),
                                   low=cutoffs["low"], high=cutoffs["high"])
    table = labels.merge(table, on=["author", "post"], how="left", validate="one_to_one")
    if table.text.isna().any():
        raise ValueError("Labelled posts missing in the release")
    if set(table.loc[table.split == "train", "author"]) & set(table.loc[table.split == "test", "author"]):
        raise ValueError("Official train/test authors overlap")
    return table, numeric


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--final", action="store_true", help="Also freeze a final model and evaluate the official test once")
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.final and (RESULTS / "test_metrics.json").exists():
        raise SystemExit("A final test evaluation already exists; preserve it instead of repeatedly tuning on test.")
    started = time.time()
    table, numeric = load_table(args.data)
    train = table.loc[table.split == "train"].reset_index(drop=True)
    authors = np.array(sorted(train.author.unique()))
    np.random.default_rng(SEED).shuffle(authors)
    folds = np.array_split(authors, args.folds)
    rows, records = [], []
    for number, validation_authors in enumerate(folds, 1):
        print(f"Outer author fold {number}/{args.folds}", flush=True)
        validation = train.loc[train.author.isin(validation_authors)]
        fitting = train.loc[~train.author.isin(validation_authors)]
        predictions, bundle = fit_predict(fitting, validation, numeric, args.iterations)
        result = validation[["author", "post", "label", "content_type", "hist_n"]].copy()
        for name, predicted in predictions.items():
            result[name] = predicted
        result["fold"] = number
        rows.append(result)
        record = dict(fold=number, config=bundle["config"],
                      validation_authors=sorted(validation_authors.tolist()),
                      fitting_authors=bundle["fitting_authors"],
                      calibration_authors=bundle["calibration_authors"],
                      metrics={name: metrics(validation.label, p) for name, p in predictions.items()})
        records.append(record)
        (RESULTS / f"fold_{number}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print({name: record["metrics"][name]["macro_f1"] for name in predictions}, flush=True)
    oof = pd.concat(rows, ignore_index=True)
    oof.to_csv(RESULTS / "oof_predictions.csv", index=False)
    summary = dict(seed=SEED, folds=records, iterations=args.iterations,
        oof={name: metrics(oof.label, oof[name]) for name in predictions},
        fusion_author_bootstrap_95=bootstrap_authors(oof, "fusion"),
        fusion_minus_history_95=bootstrap_authors(oof, "fusion", "history"),
        by_content_type={str(kind): metrics(group.label, group.fusion)
                         for kind, group in oof.groupby("content_type") if len(group) >= 30})
    (RESULTS / "validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.final:
        print("Freezing the final model; official test evaluation follows.", flush=True)
        test = table.loc[table.split == "test"].reset_index(drop=True)
        predicted, bundle = fit_predict(train, test.drop(columns="label"), numeric,
                                        args.iterations, snapshot_ablation=False)
        joblib.dump(bundle, RESULTS / "model.joblib")
        (RESULTS / "frozen_config.json").write_text(json.dumps({
            "seed": SEED, "iterations": args.iterations, "config": bundle["config"],
            "numeric": numeric, "fitting_authors": bundle["fitting_authors"],
            "calibration_authors": bundle["calibration_authors"],
            "n_fitting": bundle["n_fitting"], "n_calibration": bundle["n_calibration"],
        }, indent=2), encoding="utf-8")
        final = test[["author", "post"]].assign(label=predicted["fusion"])
        (ROOT / "results" / "predictions").mkdir(parents=True, exist_ok=True)
        final.to_csv(ROOT / "results" / "predictions" / "maria.csv", index=False)
        measured = metrics(test.label, predicted["fusion"])
        measured["author_bootstrap_95"] = bootstrap_authors(test.assign(predicted=predicted["fusion"]), "predicted")
        (RESULTS / "test_metrics.json").write_text(json.dumps(measured, indent=2), encoding="utf-8")
        print(f"Final test macro-F1: {measured['macro_f1']}", flush=True)
    print(f"Finished in {time.time() - started:.0f}s. Results: {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
