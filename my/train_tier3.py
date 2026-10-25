"""Tier 3 training: Tier 1 + Tier 2 + CLIP image features."""

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
from my import features_tier3 as t3
from my.train_tier1 import (_lgbm_cat, train_lightgbm, top_features,
                             SEED, N_SPLITS, RESULTS)


def build_table():
    labels = load_labels(DATA_DIR)
    tier1 = t1.materialise(DATA_DIR)
    tier2 = t2.materialise()
    tier3 = t3.materialise()
    merged = (tier1
              .merge(tier2, on=["author", "post"], how="left", validate="one_to_one")
              .merge(tier3, on=["author", "post"], how="left", validate="one_to_one"))
    table = labels.merge(merged, on=["author", "post"], how="left",
                         validate="one_to_one")
    numeric = t1.TIER1_NUMERIC + t2.TIER2_NUMERIC + t3.TIER3_NUMERIC
    cat = t1.TIER1_CAT
    for c in numeric:
        table[c] = pd.to_numeric(table[c], errors="coerce").astype(float).fillna(0.0)
    table[cat] = table[cat].fillna("unknown").astype(str)
    return table, numeric, cat


def main():
    t0 = time.time()
    print("loading features...", flush=True)
    table, numeric, cat = build_table()
    print(f"  table: {len(table):,} rows   features: {len(numeric)+len(cat)}",
          flush=True)

    train = table[table["split"] == "train"].reset_index(drop=True)
    test = table[table["split"] == "test"].reset_index(drop=True)
    y = train["label"].values.astype(int)
    groups = train["author"].values
    X = train[numeric + cat]
    X_te = test[numeric + cat]

    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_pred = np.full(len(train), -1, dtype=int)
    folds = []
    print("\n=== LIGHTGBM Tier1+Tier2+Tier3 ===", flush=True)
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        X_tr_c, X_val_c = _lgbm_cat(X_tr, X_val)
        clf = train_lightgbm(X_tr_c, y[tr_idx], X_val_c, y[val_idx])
        pred = clf.predict_proba(X_val_c).argmax(axis=1)
        oof_pred[val_idx] = pred
        m = score(y[val_idx], pred)
        folds.append(m)
        print(f"  fold {fold}: macro-F1={m['macro_f1']:.4f}  QWK={m['qwk']:.4f}  "
              f"acc={m['accuracy']:.4f}  (n_val={len(val_idx):,})", flush=True)
    oof = score(y, oof_pred)
    print(f"  OOF overall: macro-F1={oof['macro_f1']:.4f}  QWK={oof['qwk']:.4f}  "
          f"acc={oof['accuracy']:.4f}", flush=True)

    # Refit full
    print("\n--- Refit, predict official test ---", flush=True)
    uniq = train["author"].drop_duplicates().tolist()
    val_auth = set(uniq[int(len(uniq) * 0.85):])
    val_mask = train["author"].isin(val_auth).values
    X_fit_c, X_val_c = _lgbm_cat(X[~val_mask], X[val_mask])
    _, X_te_c = _lgbm_cat(X_fit_c, X_te)
    clf = train_lightgbm(X_fit_c, y[~val_mask], X_val_c, y[val_mask])
    proba = clf.predict_proba(X_te_c)
    pred = proba.argmax(axis=1)
    tm = score(test["label"].values.astype(int), pred)
    print(f"  test: macro-F1={tm['macro_f1']:.4f}  QWK={tm['qwk']:.4f}  "
          f"acc={tm['accuracy']:.4f}", flush=True)

    top = top_features(clf, (numeric + cat))
    print("  top features:")
    for n, v in top[:20]:
        print(f"    {n:30s} {v:10.3f}")

    out_path = RESULTS / "predictions" / "tier3_lightgbm.csv"
    test[["author", "post"]].assign(label=pred.astype(int)).to_csv(out_path, index=False)
    print(f"  wrote {out_path}", flush=True)

    (RESULTS / "tier3_cv.json").write_text(json.dumps({
        "seed": SEED, "n_splits": N_SPLITS,
        "metrics": {"lightgbm": {"oof": oof, "folds": folds, "test": tm,
                                  "top_features": top}}
    }, indent=2), encoding="utf-8")
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
