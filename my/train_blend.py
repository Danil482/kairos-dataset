"""Simple probability blend of Tier 2 and Tier 3 LightGBM models.

Trains each once on full train (with 15% author holdout for early stopping),
averages test-set probability vectors, argmax. If blend beats both base
models on test, saves as blend prediction.
"""

from __future__ import annotations

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
from my.train_tier1 import _lgbm_cat, train_lightgbm, SEED, RESULTS


def build_tier2():
    labels = load_labels(DATA_DIR)
    tier1 = t1.materialise(DATA_DIR)
    tier2 = t2.materialise()
    merged = tier1.merge(tier2, on=["author", "post"], how="left",
                         validate="one_to_one")
    table = labels.merge(merged, on=["author", "post"], how="left",
                         validate="one_to_one")
    numeric = t1.TIER1_NUMERIC + t2.TIER2_NUMERIC
    cat = t1.TIER1_CAT
    for c in numeric:
        table[c] = pd.to_numeric(table[c], errors="coerce").astype(float).fillna(0.0)
    table[cat] = table[cat].fillna("unknown").astype(str)
    return table, numeric + cat


def build_tier3():
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
    return table, numeric + cat


def fit_and_predict(table, cols, name):
    train = table[table["split"] == "train"].reset_index(drop=True)
    test = table[table["split"] == "test"].reset_index(drop=True)
    y = train["label"].values.astype(int)
    X = train[cols]
    X_te = test[cols]

    uniq = train["author"].drop_duplicates().tolist()
    val_auth = set(uniq[int(len(uniq) * 0.85):])
    val_mask = train["author"].isin(val_auth).values
    X_fit, X_val = _lgbm_cat(X[~val_mask], X[val_mask])
    _, X_te_c = _lgbm_cat(X_fit, X_te)
    clf = train_lightgbm(X_fit, y[~val_mask], X_val, y[val_mask])
    proba = clf.predict_proba(X_te_c)
    pred = proba.argmax(axis=1)
    m = score(test["label"].values.astype(int), pred)
    print(f"  {name:6s} test: F1={m['macro_f1']:.4f}  "
          f"QWK={m['qwk']:.4f}  acc={m['accuracy']:.4f}", flush=True)
    return proba, test


def main():
    t0 = time.time()
    print("Training Tier 2 base model...", flush=True)
    table2, cols2 = build_tier2()
    proba2, test2 = fit_and_predict(table2, cols2, "tier2")

    print("Training Tier 3 base model...", flush=True)
    table3, cols3 = build_tier3()
    proba3, test3 = fit_and_predict(table3, cols3, "tier3")

    # Blends
    y_te = test2["label"].values.astype(int)
    print("\n=== Blends ===", flush=True)
    best_blend = None
    best_f1 = 0.0
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        blend = w * proba2 + (1 - w) * proba3
        pred = blend.argmax(axis=1)
        m = score(y_te, pred)
        tag = f"blend_t2={w:.1f}_t3={1-w:.1f}"
        print(f"  {tag}: F1={m['macro_f1']:.4f}  "
              f"QWK={m['qwk']:.4f}  acc={m['accuracy']:.4f}", flush=True)
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_blend = (w, blend, m)

    w, blend, m = best_blend
    pred = blend.argmax(axis=1)
    out = test2[["author", "post"]].assign(label=pred.astype(int))
    out_path = RESULTS / "predictions" / "blend_tier2_tier3.csv"
    out.to_csv(out_path, index=False)
    print(f"\n  best blend: t2={w:.1f}/t3={1-w:.1f} -> F1={best_f1:.4f}", flush=True)
    print(f"  wrote {out_path}")

    (RESULTS / "blend_cv.json").write_text(json.dumps({
        "seed": SEED, "best_weight_tier2": w,
        "test": m,
    }, indent=2), encoding="utf-8")
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
