"""Train lecture classifiers on saved author folds; never open official test labels.

TF-IDF + LinearSVC, TF-IDF + MultinomialNB, tabular Random Forest,
and native-text CatBoost with inner early stopping. All transforms are fitted
only on fitting authors. Sigmoid calibration, stopping and fusion selection use
the inner calibration authors; outer validation is used only for reporting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "6")

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

from train_maria import CATEGORY, SEED, bootstrap_authors, metrics, text_model
from scripts.pilot_models import calibrate_single, restrict_support

SOURCE = ROOT / "results" / "exploration_v2"
OUT = ROOT / "results" / "lecture_models"
NAMES = ["linear_svm", "naive_bayes", "random_forest", "catboost_800", "catboost_svm_fusion"]


def split_by_plan(table, plan):
    if set(table.split) != {"train"} or table.duplicated(["author", "post"]).any():
        raise ValueError("Expected unique official train posts only")
    groups = [set(plan[key]) for key in ("fitting_authors", "calibration_authors", "validation_authors")]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("Fitting/calibration/validation authors overlap")
    if set.union(*groups) != set(table.author):
        raise ValueError("Author plan does not cover the cached train population")
    return tuple(table.loc[table.author.isin(group)].copy() for group in groups)


def expanded_text(table):
    values = table.expanded_text.fillna("").str.slice(0, 20000)
    return values.mask(values.str.strip().eq(""), "empty_post")


def ordered_probabilities(model, matrix):
    if list(model.classes_) != [0, 1, 2]:
        raise ValueError("Probability columns must have below/typical/above order")
    values = model.predict_proba(matrix)
    if not np.isfinite(values).all() or not np.allclose(values.sum(axis=1), 1):
        raise ValueError("Invalid probability matrix")
    return values


def choose_fusion(labels, boost, svm):
    best = None
    for weight in (0., .1, .25, .5):
        config = calibrate_single(labels, (1 - weight) * boost + weight * svm)
        config["svm_weight"] = weight
        if best is None or config["calibration_macro_f1"] > best["calibration_macro_f1"]:
            best = config
    return best


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def train_fold(number, payload, fingerprint):
    directory = OUT / f"fold_{number}"
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "report.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("complete"):
            if previous.get("input_sha256") != fingerprint:
                raise ValueError("Input cache changed; preserve the completed experiment")
            print(f"Fold {number} already complete; preserving saved models", flush=True)
            return
    plan = json.loads((ROOT / "results" / "maria" / f"fold_{number}.json").read_text())
    fitting, calibration, validation = split_by_plan(payload["table"], plan)
    numeric = payload["numeric"] + payload["extra_numeric"]
    cats = CATEGORY + ["writing_system"]
    columns = numeric + cats
    forbidden = {"author", "post", "label", "class", "split", "reactions", "resid", "age_days", "log_followers"}
    if forbidden.intersection(columns) or len(set(columns)) != len(columns):
        raise ValueError("Invalid modeling feature allowlist")
    for part in (fitting, calibration, validation):
        part["expanded_text"] = expanded_text(part)
    report = dict(fold=number, seed=SEED, input_sha256=fingerprint, complete=False,
                  scope="Development CV on official train authors; not an official test score",
                  fitting_posts=len(fitting), calibration_posts=len(calibration), validation_posts=len(validation),
                  models={})
    # Keep only the split membership, not unrelated previous model results.
    report["author_plan"] = {key:plan[key] for key in ("fitting_authors", "calibration_authors", "validation_authors")}
    predictions = validation[["author", "post", "label"]].copy()
    calibration[["author", "post", "label"]].to_csv(directory / "calibration_keys.csv", index=False)
    probabilities = {}
    low = payload["cutoffs"][0]
    save_json(report_path, report)

    def record(name, raw_cal, raw_val, seconds, details=None, decision=None):
        cal = restrict_support(raw_cal, calibration, low)
        val = restrict_support(raw_val, validation, low)
        config = decision or calibrate_single(calibration.label, cal)
        predicted = np.argmax(val * config["multipliers"], axis=1)
        predictions[name] = predicted
        probabilities[name] = (cal, val)
        result = dict(metrics=metrics(validation.label, predicted), calibration=config,
                      seconds=round(seconds, 1), details=details or {})
        report["models"][name] = result
        np.savez_compressed(directory / f"{name}_probabilities.npz", calibration=cal, validation=val)
        predictions.to_csv(directory / "predictions.csv", index=False)
        save_json(report_path, report)
        print(f"Fold {number} {name}: macro-F1={result['metrics']['macro_f1']:.4f}; {seconds:.1f}s", flush=True)

    print(f"Fold {number}: fitting word and character TF-IDF on {len(fitting)} posts", flush=True)
    started = time.monotonic()
    vectorizer = text_model().named_steps["tfidf"]
    train_matrix = vectorizer.fit_transform(fitting.expanded_text)
    cal_matrix = vectorizer.transform(calibration.expanded_text)
    val_matrix = vectorizer.transform(validation.expanded_text)
    joblib.dump(vectorizer, directory / "tfidf.joblib", compress=3)
    report["tfidf"] = dict(features=train_matrix.shape[1], seconds=round(time.monotonic()-started, 1),
                            word_ngram_range=[1, 2], char_ngram_range=[3, 5], expanded_text_cap=20000)
    save_json(report_path, report)
    print(f"Fold {number}: fitting LinearSVC", flush=True)
    started = time.monotonic()
    svm = LinearSVC(C=1., class_weight="balanced", random_state=SEED, max_iter=10000, tol=1e-4)
    svm.fit(train_matrix, fitting.label)
    calibrated = CalibratedClassifierCV(FrozenEstimator(svm), method="sigmoid")
    calibrated.fit(cal_matrix, calibration.label)
    joblib.dump(calibrated, directory / "linear_svm.joblib", compress=3)
    record("linear_svm", ordered_probabilities(calibrated, cal_matrix), ordered_probabilities(calibrated, val_matrix),
           time.monotonic()-started, dict(C=1., class_weight="balanced", iterations=int(svm.n_iter_),
                                        probability_calibration="sigmoid on separate inner authors"))
    started = time.monotonic()
    bayes = MultinomialNB(alpha=1.)
    bayes.fit(train_matrix, fitting.label)
    joblib.dump(bayes, directory / "naive_bayes.joblib", compress=3)
    record("naive_bayes", ordered_probabilities(bayes, cal_matrix), ordered_probabilities(bayes, val_matrix),
           time.monotonic()-started, dict(alpha=1.))
    del train_matrix, cal_matrix, val_matrix, vectorizer, calibrated, svm, bayes

    print(f"Fold {number}: fitting 300 Random Forest trees", flush=True)
    started = time.monotonic()
    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
    ])
    forest = Pipeline([
        ("features", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=10,
            max_features=.7, class_weight="balanced_subsample", n_jobs=6, random_state=SEED)),
    ])
    forest.fit(fitting[columns], fitting.label)
    joblib.dump(forest, directory / "random_forest.joblib", compress=3)
    record("random_forest", ordered_probabilities(forest, calibration[columns]),
           ordered_probabilities(forest, validation[columns]), time.monotonic()-started,
           dict(trees=300, depth=20, min_samples_leaf=10, max_features=.7, class_weight="balanced_subsample"))
    del forest

    print(f"Fold {number}: fitting CatBoost, up to 800 trees; stopping uses inner authors", flush=True)
    started = time.monotonic()
    columns = columns + ["expanded_text"]
    model = CatBoostClassifier(iterations=800, depth=6, learning_rate=.045,
        loss_function="MultiClass", l2_leaf_reg=5, random_seed=SEED, thread_count=6,
        verbose=False, allow_writing_files=False,
        feature_calcers=["BoW:top_tokens_count=2000", "NaiveBayes", "BM25"])
    model.fit(fitting[columns], fitting.label, cat_features=cats, text_features=["expanded_text"],
              eval_set=(calibration[columns], calibration.label), early_stopping_rounds=100, use_best_model=True)
    model.save_model(str(directory / "catboost_800.cbm"))
    save_json(directory / "feature_schema.json", dict(numeric=numeric, categorical=cats,
              catboost_columns=columns, cutoffs=payload["cutoffs"], class_order=["below", "typical", "above"]))
    record("catboost_800", ordered_probabilities(model, calibration[columns]),
           ordered_probabilities(model, validation[columns]), time.monotonic()-started,
           dict(max_iterations=800, trees=int(model.tree_count_), depth=6, learning_rate=.045,
                stopping_metric="MultiClass on inner calibration authors", patience=100))
    started = time.monotonic()
    boost_cal, boost_val = probabilities["catboost_800"]
    svm_cal, svm_val = probabilities["linear_svm"]
    fusion = choose_fusion(calibration.label, boost_cal, svm_cal)
    weight = fusion["svm_weight"]
    record("catboost_svm_fusion", (1-weight)*boost_cal+weight*svm_cal,
           (1-weight)*boost_val+weight*svm_val, time.monotonic()-started,
           dict(svm_weight_candidates=[0., .1, .25, .5]), decision=fusion)
    report["complete"] = True
    save_json(report_path, report)


def summarize(payload):
    parts, folds = [], []
    for number in (1, 2, 3):
        path = OUT / f"fold_{number}" / "report.json"
        if not path.exists():
            return
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report["complete"]:
            return
        folds.append(report)
        parts.append(pd.read_csv(path.parent / "predictions.csv").assign(fold=number))
    combined = pd.concat(parts, ignore_index=True)
    expected = payload["table"][["author", "post", "label"]]
    coverage = combined.merge(expected, on=["author", "post", "label"], how="outer",
                              indicator=True, validate="one_to_one")
    if not coverage._merge.eq("both").all():
        raise ValueError("Outer predictions do not cover each train post exactly once")
    reference = pd.read_csv(SOURCE / "all_oof_predictions.csv")
    combined = combined.merge(reference[["author", "post", "label", "native_text_unweighted", "v1_fusion"]],
                              on=["author", "post", "label"], validate="one_to_one", how="left")
    if combined.isna().any().any():
        raise ValueError("Previous development predictions do not match")
    models = NAMES + ["native_text_unweighted", "v1_fusion"]
    report = dict(n=len(combined), authors=int(combined.author.nunique()),
        scope="Exploratory development comparison on three official-train author folds. No new official test score.",
        oof={name:metrics(combined.label, combined[name]) for name in models},
        folds={str(f["fold"]):{name:f["models"][name]["metrics"]["macro_f1"] for name in NAMES} for f in folds},
        differences_vs_previous_catboost={name:bootstrap_authors(combined, name, "native_text_unweighted")
                                         for name in ("catboost_800", "catboost_svm_fusion", "random_forest")},
        interpretation="Intervals resample authors and condition on fitted predictions; they do not include model-selection uncertainty.")
    report["best_development_model"] = max(models, key=lambda name:f1_score(combined.label, combined[name], average="macro"))
    combined.to_csv(OUT / "oof_predictions.csv", index=False)
    save_json(OUT / "comparison.json", report)
    print(json.dumps({"oof_macro_f1":{name:report["oof"][name]["macro_f1"] for name in models},
                      "best_development_model":report["best_development_model"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=[1, 2, 3], help="Default: all three folds, sequentially")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    source = SOURCE / "train_table.joblib"
    with source.open("rb") as stream:
        fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
    payload = joblib.load(source)
    if not args.summarize_only:
        for number in ([args.fold] if args.fold else [1, 2, 3]):
            train_fold(number, payload, fingerprint)
    summarize(payload)


if __name__ == "__main__":
    main()
