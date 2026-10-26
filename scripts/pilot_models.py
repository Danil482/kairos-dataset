"""Controlled train-only pilot; leaves the frozen v1 model and official test alone.

Use --fold 1, 2 or 3 to reproduce the saved outer author folds. A one-fold
pilot is exploratory evidence, not an independent estimate of the winner.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "6")

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

from train_maria import CATEGORY, SEED, metrics, bootstrap_authors

OUT = ROOT / "results" / "exploration_v2"


def calibrate_single(y, probability):
    best = None
    for below in (.6, .8, 1., 1.2, 1.4, 1.6):
        for above in (.6, .8, 1., 1.2, 1.4, 1.6):
            multipliers = [below, 1., above]
            score = f1_score(y, np.argmax(probability * multipliers, axis=1), average="macro", labels=[0, 1, 2])
            if best is None or score > best[0]:
                best = (float(score), multipliers)
    return dict(calibration_macro_f1=best[0], multipliers=best[1])


def restrict_support(probability, table, low):
    probability = probability.copy()
    impossible = (table.level <= -low) & (table.timestamp_tie == 0)
    probability[impossible.to_numpy(), 0] = 0.
    return probability / probability.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=[1, 2, 3], default=1)
    args = parser.parse_args()
    payload = joblib.load(OUT / "train_table.joblib")
    table, numeric = payload["table"], payload["numeric"]
    plan = json.loads((ROOT / "results" / "maria" / f"fold_{args.fold}.json").read_text())
    fitting = table.loc[table.author.isin(plan["fitting_authors"])].copy()
    calibration = table.loc[table.author.isin(plan["calibration_authors"])].copy()
    validation = table.loc[table.author.isin(plan["validation_authors"])].copy()
    assert set(table.split) == {"train"}
    assert not set(fitting.author) & set(calibration.author)
    assert not set(fitting.author) & set(validation.author)
    assert not set(calibration.author) & set(validation.author)
    for part in (fitting, calibration, validation):
        part["expanded_text"] = part.expanded_text.str.slice(0, 20000).mask(part.expanded_text.str.strip().eq(""), "empty_post")
    cases = [
        ("history_balanced", False, False, "Balanced"),
        ("history_unweighted", False, False, None),
        ("enriched_unweighted", True, False, None),
        ("native_text_unweighted", True, True, None),
    ]
    report = dict(fold=args.fold, seed=SEED, scope="one saved outer fold of official train authors",
                  fitting_posts=len(fitting), calibration_posts=len(calibration), validation_posts=len(validation),
                  current_fusion_reference=plan["metrics"]["fusion"], models={})
    predictions = validation[["author", "post", "label"]].copy()
    low = payload["cutoffs"][0]
    for name, enriched, text, weighting in cases:
        started = time.time()
        print(f"Fitting {name}...", flush=True)
        cats = CATEGORY + (["writing_system"] if enriched else [])
        columns = numeric + (payload["extra_numeric"] if enriched else []) + cats
        if text:
            columns += ["expanded_text"]
        options = dict(iterations=400, depth=6, learning_rate=.06, loss_function="MultiClass",
                       l2_leaf_reg=5, random_seed=SEED, thread_count=6, verbose=False,
                       allow_writing_files=False)
        if weighting:
            options["auto_class_weights"] = weighting
        if text:
            options["feature_calcers"] = ["BoW:top_tokens_count=2000", "NaiveBayes", "BM25"]
        model = CatBoostClassifier(**options)
        model.fit(fitting[columns], fitting.label, cat_features=cats,
                  text_features=["expanded_text"] if text else None)
        raw_cal = model.predict_proba(calibration[columns])
        raw_val = model.predict_proba(validation[columns])
        probability = restrict_support(raw_cal, calibration, low)
        values = restrict_support(raw_val, validation, low)
        config = calibrate_single(calibration.label, probability)
        predicted = np.argmax(values * config["multipliers"], axis=1)
        predictions[name] = predicted
        plain_config = calibrate_single(calibration.label, raw_cal)
        plain = np.argmax(raw_val * plain_config["multipliers"], axis=1)
        feature_importance = sorted([dict(feature=c, importance=float(v))
            for c,v in zip(columns, model.feature_importances_)], key=lambda x:-x["importance"])
        result = dict(metrics=metrics(validation.label, predicted),
                      without_support_constraint=metrics(validation.label, plain),
                      calibration=config, seconds=round(time.time() - started, 1),
                      feature_importance=feature_importance)
        report["models"][name] = result
        np.savez_compressed(OUT / f"pilot_fold{args.fold}_{name}_probabilities.npz",
                            calibration=probability, validation=values)
        (OUT / f"pilot_fold{args.fold}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        predictions.to_csv(OUT / f"pilot_fold{args.fold}_predictions.csv", index=False)
        print(name, {k:result["metrics"][k] for k in ["macro_f1", "qwk", "accuracy"]},
              "seconds", result["seconds"], flush=True)
    report["native_minus_enriched_95"] = bootstrap_authors(predictions, "native_text_unweighted", "enriched_unweighted")
    report["enriched_minus_history_95"] = bootstrap_authors(predictions, "enriched_unweighted", "history_unweighted")
    (OUT / f"pilot_fold{args.fold}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
