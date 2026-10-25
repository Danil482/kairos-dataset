"""Train CatBoost and LightGBM on Tier 1 features with GroupKFold(5) by author.

Reports OOF macro-F1 / QWK / accuracy per fold and overall. Picks the best
model by OOF macro-F1 and writes:
    results/predictions/tier1_<model>.csv  (predictions on the official test split)
    results/tier1_cv.json                   (OOF metrics + fold detail)
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

SEED = 42
N_SPLITS = 5
RESULTS = Path(__file__).resolve().parent.parent / "results"


def build_table(data_dir: str | Path) -> pd.DataFrame:
    labels = load_labels(data_dir)
    feats = t1.materialise(data_dir)
    table = labels.merge(feats, on=["author", "post"],
                         how="left", validate="one_to_one")
    if table["level"].isna().any():
        raise SystemExit("some labelled posts have no features: release incomplete?")
    return table


def _prep_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = t1.TIER1_NUMERIC + t1.TIER1_CAT
    X = df[cols].copy()
    # fill any residual NaNs so sklearn-side models don't complain
    for c in t1.TIER1_NUMERIC:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    X[t1.TIER1_NUMERIC] = X[t1.TIER1_NUMERIC].fillna(0.0)
    X[t1.TIER1_CAT] = X[t1.TIER1_CAT].fillna("unknown").astype(str)
    return X


def train_catboost(X_tr, y_tr, X_val, y_val):
    from catboost import CatBoostClassifier
    cat_idx = [X_tr.columns.get_loc(c) for c in t1.TIER1_CAT]
    counts = np.bincount(y_tr, minlength=3).astype(float)
    class_weights = (counts.sum() / (3 * np.clip(counts, 1, None))).tolist()
    clf = CatBoostClassifier(
        iterations=800,
        learning_rate=0.06,
        depth=7,
        loss_function="MultiClass",
        class_weights=class_weights,
        cat_features=cat_idx,
        random_seed=SEED,
        od_type="Iter",
        od_wait=60,
        verbose=False,
        thread_count=-1,
    )
    clf.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
    return clf


def _lgbm_cat(X_tr: pd.DataFrame, X_val: pd.DataFrame | None = None):
    X_tr = X_tr.copy()
    for c in t1.TIER1_CAT:
        X_tr[c] = X_tr[c].astype("category")
    if X_val is None:
        return X_tr, None
    X_val = X_val.copy()
    for c in t1.TIER1_CAT:
        X_val[c] = pd.Categorical(X_val[c], categories=X_tr[c].cat.categories)
    return X_tr, X_val


def train_lightgbm(X_tr, y_tr, X_val, y_val):
    import lightgbm as lgb
    counts = np.bincount(y_tr, minlength=3).astype(float)
    class_weight = {i: counts.sum() / (3 * max(counts[i], 1)) for i in range(3)}
    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=1500,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=SEED,
        class_weight=class_weight,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(80, verbose=False),
                       lgb.log_evaluation(0)])
    return clf


def oof_cv(table: pd.DataFrame, model_name: str):
    train = table[table["split"] == "train"].reset_index(drop=True)
    y = train["label"].values.astype(int)
    groups = train["author"].values
    X = _prep_matrix(train)

    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_pred = np.full(len(train), -1, dtype=int)
    oof_proba = np.zeros((len(train), 3))
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
        oof_proba[val_idx] = p
        m = score(y_val, pred)
        fold_scores.append(m)
        print(f"  fold {fold}: macro-F1={m['macro_f1']:.4f}  QWK={m['qwk']:.4f}  "
              f"acc={m['accuracy']:.4f}  (n_val={len(val_idx):,})",
              flush=True)

    oof = score(y, oof_pred)
    print(f"  OOF overall: macro-F1={oof['macro_f1']:.4f}  QWK={oof['qwk']:.4f}  "
          f"acc={oof['accuracy']:.4f}",
          flush=True)
    return oof, fold_scores, oof_pred, oof_proba


def fit_full_and_predict(table: pd.DataFrame, model_name: str):
    train = table[table["split"] == "train"].reset_index(drop=True)
    test = table[table["split"] == "test"].reset_index(drop=True)
    X_tr = _prep_matrix(train)
    X_te = _prep_matrix(test)
    y_tr = train["label"].values.astype(int)
    # pseudo validation = last 15% of train authors (deterministic by order)
    uniq_authors = train["author"].drop_duplicates().tolist()
    val_authors = set(uniq_authors[int(len(uniq_authors) * 0.85):])
    val_mask = train["author"].isin(val_authors).values
    if model_name == "catboost":
        clf = train_catboost(X_tr[~val_mask], y_tr[~val_mask],
                             X_tr[val_mask], y_tr[val_mask])
        proba = clf.predict_proba(X_te)
    else:
        X_tr_fit, X_val_fit = _lgbm_cat(X_tr[~val_mask], X_tr[val_mask])
        clf = train_lightgbm(X_tr_fit, y_tr[~val_mask],
                             X_val_fit, y_tr[val_mask])
        _, X_te_c = _lgbm_cat(X_tr_fit, X_te)
        proba = clf.predict_proba(X_te_c)
    pred = proba.argmax(axis=1)
    test_metrics = score(test["label"].values.astype(int), pred)
    print(f"  test: macro-F1={test_metrics['macro_f1']:.4f}  "
          f"QWK={test_metrics['qwk']:.4f}  acc={test_metrics['accuracy']:.4f}",
          flush=True)
    return clf, test, pred, proba, test_metrics


def top_features(clf, feature_names, k: int = 15):
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    else:
        imp = clf.get_feature_importance()
    order = np.argsort(imp)[::-1][:k]
    return [(feature_names[i], float(imp[i])) for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DIR)
    args = ap.parse_args()
    t0 = time.time()

    print("loading data and features...", flush=True)
    table = build_table(args.data)
    print(f"  table: {len(table):,} rows, {table.shape[1]} columns", flush=True)

    all_metrics = {}
    for model_name in ["catboost", "lightgbm"]:
        print(f"\n=== {model_name.upper()} — GroupKFold({N_SPLITS}) by author ===", flush=True)
        oof, fold_scores, oof_pred, oof_proba = oof_cv(table, model_name)
        print(f"\n--- Refit on full train, predict official test ---", flush=True)
        clf, test, pred, proba, test_metrics = fit_full_and_predict(table, model_name)

        top = top_features(clf, _prep_matrix(table).columns.tolist())
        print(f"  top features:")
        for n, v in top:
            print(f"    {n:30s} {v:10.3f}")

        all_metrics[model_name] = {
            "oof": oof,
            "folds": fold_scores,
            "test": test_metrics,
            "top_features": top,
        }

        # write prediction CSV in the format evaluate.py expects
        out = test[["author", "post"]].assign(label=pred.astype(int))
        out_path = RESULTS / "predictions" / f"tier1_{model_name}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"  wrote {out_path}", flush=True)

        # checkpoint JSON after each model so a crash in later ones does not
        # lose the earlier results
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "tier1_cv.json").write_text(json.dumps({
            "seed": SEED, "n_splits": N_SPLITS,
            "metrics": all_metrics,
        }, indent=2), encoding="utf-8")

    # overall summary
    best = max(all_metrics.items(), key=lambda kv: kv[1]["oof"]["macro_f1"])
    print(f"\nBest by OOF macro-F1: {best[0]} ({best[1]['oof']['macro_f1']:.4f})")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "tier1_cv.json").write_text(json.dumps({
        "seed": SEED, "n_splits": N_SPLITS,
        "metrics": all_metrics,
        "best": best[0],
    }, indent=2), encoding="utf-8")
    print(f"\nwrote results/tier1_cv.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
