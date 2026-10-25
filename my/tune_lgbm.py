"""Optuna hyperparameter search for the LightGBM multiclass model.

Objective: maximise 5-fold GroupKFold-by-author OOF macro-F1 on the Tier 2
feature table. ~25 trials, each one evaluating 5 folds with early stopping.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from kairos.data import DATA_DIR, load_labels
from kairos.metrics import score

from my import features_tier1 as t1
from my import features_tier2 as t2
from my.train_tier1 import _lgbm_cat, SEED, N_SPLITS, RESULTS


def build_table():
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
    return table, numeric, cat


def _lgb_train(params, X_tr, y_tr, X_val, y_val, counts):
    import lightgbm as lgb
    class_weight = {i: counts.sum() / (3 * max(counts[i], 1)) for i in range(3)}
    clf = lgb.LGBMClassifier(
        objective="multiclass", num_class=3,
        n_estimators=2500,
        random_state=SEED,
        class_weight=class_weight,
        n_jobs=-1,
        verbose=-1,
        **params,
    )
    clf.fit(X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(80, verbose=False),
                       lgb.log_evaluation(0)])
    return clf


def objective(trial, table, numeric, cat):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 191),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 120),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 2.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "subsample_freq": 1,
    }

    train = table[table["split"] == "train"].reset_index(drop=True)
    y = train["label"].values.astype(int)
    groups = train["author"].values
    X = train[numeric + cat]

    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_pred = np.full(len(train), -1, dtype=int)
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        X_tr_c, X_val_c = _lgbm_cat(X_tr, X_val)
        counts = np.bincount(y[tr_idx], minlength=3).astype(float)
        clf = _lgb_train(params, X_tr_c, y[tr_idx], X_val_c, y[val_idx], counts)
        pred = clf.predict_proba(X_val_c).argmax(axis=1)
        oof_pred[val_idx] = pred
        # pruning
        trial.report(score(y[val_idx], pred)["macro_f1"], step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
    oof = score(y, oof_pred)
    return oof["macro_f1"]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=25)
    ap.add_argument("--timeout", type=int, default=5400)  # 90 min cap
    args = ap.parse_args()
    t0 = time.time()

    table, numeric, cat = build_table()
    print(f"loaded features: {len(table):,} rows, "
          f"{len(numeric) + len(cat)} features", flush=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )

    def _print_trial(study, trial):
        print(f"trial {trial.number:3d}: value={trial.value:.4f}  "
              f"best={study.best_value:.4f}  state={trial.state.name}",
              flush=True)

    study.optimize(lambda t: objective(t, table, numeric, cat),
                   n_trials=args.n_trials,
                   timeout=args.timeout,
                   callbacks=[_print_trial],
                   show_progress_bar=False)

    print(f"\nbest OOF macro-F1 = {study.best_value:.4f}")
    print(f"best params: {study.best_params}")

    # Refit best on full train, predict test
    print("\n=== Refit best model, predict test ===", flush=True)
    train = table[table["split"] == "train"].reset_index(drop=True)
    test = table[table["split"] == "test"].reset_index(drop=True)
    y = train["label"].values.astype(int)
    X = train[numeric + cat]
    X_te = test[numeric + cat]
    uniq = train["author"].drop_duplicates().tolist()
    val_auth = set(uniq[int(len(uniq) * 0.85):])
    val_mask = train["author"].isin(val_auth).values
    X_fit_c, X_val_c = _lgbm_cat(X[~val_mask], X[val_mask])
    _, X_te_c = _lgbm_cat(X_fit_c, X_te)
    counts = np.bincount(y[~val_mask], minlength=3).astype(float)
    clf = _lgb_train(study.best_params, X_fit_c, y[~val_mask],
                     X_val_c, y[val_mask], counts)
    proba = clf.predict_proba(X_te_c)
    pred = proba.argmax(axis=1)
    tm = score(test["label"].values.astype(int), pred)
    print(f"  test: macro-F1={tm['macro_f1']:.4f}  QWK={tm['qwk']:.4f}  "
          f"acc={tm['accuracy']:.4f}", flush=True)

    out = test[["author", "post"]].assign(label=pred.astype(int))
    out_path = RESULTS / "predictions" / "tier4_lgbm_tuned.csv"
    out.to_csv(out_path, index=False)
    print(f"  wrote {out_path}")

    (RESULTS / "tier4_tune.json").write_text(json.dumps({
        "seed": SEED, "n_trials": len(study.trials),
        "best_oof_macro_f1": study.best_value,
        "best_params": study.best_params,
        "test": tm,
    }, indent=2), encoding="utf-8")
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
