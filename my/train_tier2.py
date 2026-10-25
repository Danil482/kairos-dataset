"""Tier 2 training: Tier 1 features + text-embedding derivatives.

Reuses the Tier 1 pipeline but widens the feature matrix. Trains LightGBM
(our Tier 1 winner) by default; add --with-catboost to also train CatBoost.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from kairos.data import DATA_DIR, load_labels
from kairos.metrics import score

from my import features_tier1 as t1
from my import features_tier2 as t2
from my.train_tier1 import (
    _lgbm_cat, train_catboost, train_lightgbm, top_features, SEED, N_SPLITS,
    RESULTS,
)


def build_table(data_dir: str | Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    labels = load_labels(data_dir)
    tier1 = t1.materialise(data_dir)
    tier2 = t2.materialise()
    merged = tier1.merge(tier2, on=["author", "post"],
                         how="left", validate="one_to_one")
    table = labels.merge(merged, on=["author", "post"],
                         how="left", validate="one_to_one")
    numeric = t1.TIER1_NUMERIC + t2.TIER2_NUMERIC
    cat = t1.TIER1_CAT
    for c in numeric:
        table[c] = pd.to_numeric(table[c], errors="coerce").astype(float).fillna(0.0)
    table[cat] = table[cat].fillna("unknown").astype(str)
    return table, numeric, cat


def _prep(df: pd.DataFrame, numeric, cat):
    return df[numeric + cat].copy()


def oof_cv(table: pd.DataFrame, numeric, cat, model_name: str):
    train = table[table["split"] == "train"].reset_index(drop=True)
    y = train["label"].values.astype(int)
    groups = train["author"].values
    X = _prep(train, numeric, cat)

    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_pred = np.full(len(train), -1, dtype=int)
    fold_scores = []
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        if model_name == "catboost":
            clf = train_catboost(X_tr, y_tr, X_val, y_val)
            p = clf.predict_proba(X_val)
        else:
            X_tr_c, X_val_c = _lgbm_cat(X_tr, X_val)
            clf = train_lightgbm(X_tr_c, y_tr, X_val_c, y_val)
            p = clf.predict_proba(X_val_c)
        pred = p.argmax(axis=1)
        oof_pred[val_idx] = pred
        m = score(y_val, pred)
        fold_scores.append(m)
        print(f"  fold {fold}: macro-F1={m['macro_f1']:.4f}  QWK={m['qwk']:.4f}  "
              f"acc={m['accuracy']:.4f}  (n_val={len(val_idx):,})", flush=True)

    oof = score(y, oof_pred)
    print(f"  OOF overall: macro-F1={oof['macro_f1']:.4f}  QWK={oof['qwk']:.4f}  "
          f"acc={oof['accuracy']:.4f}", flush=True)
    return oof, fold_scores


def fit_full(table: pd.DataFrame, numeric, cat, model_name: str):
    train = table[table["split"] == "train"].reset_index(drop=True)
    test = table[table["split"] == "test"].reset_index(drop=True)
    X_tr = _prep(train, numeric, cat)
    X_te = _prep(test, numeric, cat)
    y_tr = train["label"].values.astype(int)
    uniq = train["author"].drop_duplicates().tolist()
    val_authors = set(uniq[int(len(uniq) * 0.85):])
    val_mask = train["author"].isin(val_authors).values
    if model_name == "catboost":
        clf = train_catboost(X_tr[~val_mask], y_tr[~val_mask],
                             X_tr[val_mask], y_tr[val_mask])
        proba = clf.predict_proba(X_te)
    else:
        X_fit, X_val = _lgbm_cat(X_tr[~val_mask], X_tr[val_mask])
        clf = train_lightgbm(X_fit, y_tr[~val_mask], X_val, y_tr[val_mask])
        _, X_te_c = _lgbm_cat(X_fit, X_te)
        proba = clf.predict_proba(X_te_c)
    pred = proba.argmax(axis=1)
    tm = score(test["label"].values.astype(int), pred)
    print(f"  test: macro-F1={tm['macro_f1']:.4f}  QWK={tm['qwk']:.4f}  "
          f"acc={tm['accuracy']:.4f}", flush=True)
    return clf, test, pred, tm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument("--with-catboost", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    print("loading tier1 + tier2 features...", flush=True)
    table, numeric, cat = build_table(args.data)
    print(f"  table: {len(table):,} rows   features: {len(numeric) + len(cat)} "
          f"({len(numeric)} numeric + {len(cat)} cat)", flush=True)

    models = ["lightgbm"]
    if args.with_catboost:
        models.insert(0, "catboost")

    all_metrics = {}
    for m in models:
        print(f"\n=== {m.upper()} ===", flush=True)
        oof, folds = oof_cv(table, numeric, cat, m)
        print(f"\n--- Refit, predict official test ---", flush=True)
        clf, test, pred, tm = fit_full(table, numeric, cat, m)
        top = top_features(clf, (numeric + cat))
        print(f"  top features:")
        for n, v in top:
            print(f"    {n:30s} {v:10.3f}")
        all_metrics[m] = {"oof": oof, "folds": folds, "test": tm, "top_features": top}

        out = test[["author", "post"]].assign(label=pred.astype(int))
        out_path = RESULTS / "predictions" / f"tier2_{m}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"  wrote {out_path}", flush=True)

        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "tier2_cv.json").write_text(json.dumps({
            "seed": SEED, "n_splits": N_SPLITS, "metrics": all_metrics,
        }, indent=2), encoding="utf-8")

    best = max(all_metrics.items(), key=lambda kv: kv[1]["oof"]["macro_f1"])
    print(f"\nBest by OOF macro-F1: {best[0]} ({best[1]['oof']['macro_f1']:.4f})")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
