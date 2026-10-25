"""Tier 4: ordinal regression + stacking on top of the Tier 2 features.

Trains three OOF-aligned base learners with GroupKFold(5) by author:
    (a) LightGBM multiclass — our current best (Tier 2 PCA-128)
    (b) LightGBM regressor on `resid`, converted to class with frozen cutoffs
    (c) CatBoost multiclass — model-type diversity for the stacker

Then fits a Logistic Regression on the stacked OOF probabilities and uses it
to predict the official test split.

Cutoffs are the frozen train-residual percentiles that define the labels:
    low  = -0.549306
    high = +1.157453
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from kairos.data import DATA_DIR, load_labels, load_split
from kairos.metrics import score

from my import features_tier1 as t1
from my import features_tier2 as t2
from my.train_tier1 import (_lgbm_cat, train_catboost, train_lightgbm,
                             SEED, N_SPLITS, RESULTS)


def _classify(resid: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.where(resid < low, 0, np.where(resid > high, 2, 1)).astype(int)


def train_lightgbm_regressor(X_tr, y_tr, X_val, y_val):
    import lightgbm as lgb
    clf = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=2000,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(80, verbose=False),
                       lgb.log_evaluation(0)])
    return clf


def build_table(data_dir: str | Path):
    labels = load_labels(data_dir)
    tier1 = t1.materialise(data_dir)
    tier2 = t2.materialise()
    merged = tier1.merge(tier2, on=["author", "post"], how="left",
                         validate="one_to_one")
    table = labels.merge(merged, on=["author", "post"], how="left",
                         validate="one_to_one")
    # also need resid for ordinal regression
    numeric = t1.TIER1_NUMERIC + t2.TIER2_NUMERIC
    cat = t1.TIER1_CAT
    for c in numeric:
        table[c] = pd.to_numeric(table[c], errors="coerce").astype(float).fillna(0.0)
    table[cat] = table[cat].fillna("unknown").astype(str)
    return table, numeric, cat


def main():
    t0 = time.time()
    print("loading features...", flush=True)
    table, numeric, cat = build_table(DATA_DIR)
    split = load_split(DATA_DIR)
    low = split["cutoffs_on_log_residual"]["low"]
    high = split["cutoffs_on_log_residual"]["high"]
    print(f"  frozen cutoffs: low={low:.6f}  high={high:.6f}", flush=True)

    train = table[table["split"] == "train"].reset_index(drop=True)
    test = table[table["split"] == "test"].reset_index(drop=True)
    y_cls = train["label"].values.astype(int)
    y_res = train["resid"].values.astype(float)
    groups = train["author"].values
    X = train[numeric + cat]
    X_te = test[numeric + cat]

    gkf = GroupKFold(n_splits=N_SPLITS)
    n_tr = len(train)
    oof_lgb = np.zeros((n_tr, 3))
    oof_cat = np.zeros((n_tr, 3))
    oof_ord = np.zeros((n_tr, 3))   # one-hot derived from regression
    oof_ord_raw = np.zeros(n_tr)    # the raw residual prediction

    fold_scores_lgb = []
    fold_scores_cat = []
    fold_scores_ord = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_cls, groups), start=1):
        print(f"\n=== fold {fold}/{N_SPLITS}  (n_val={len(val_idx):,}) ===", flush=True)
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_cls_tr, y_cls_val = y_cls[tr_idx], y_cls[val_idx]
        y_res_tr, y_res_val = y_res[tr_idx], y_res[val_idx]

        # ------ LightGBM multiclass
        X_tr_c, X_val_c = _lgbm_cat(X_tr, X_val)
        lgb_clf = train_lightgbm(X_tr_c, y_cls_tr, X_val_c, y_cls_val)
        oof_lgb[val_idx] = lgb_clf.predict_proba(X_val_c)
        m = score(y_cls_val, oof_lgb[val_idx].argmax(axis=1))
        fold_scores_lgb.append(m)
        print(f"  lgb_mc:  F1={m['macro_f1']:.4f}  QWK={m['qwk']:.4f}  "
              f"acc={m['accuracy']:.4f}", flush=True)

        # ------ CatBoost multiclass
        cat_clf = train_catboost(X_tr, y_cls_tr, X_val, y_cls_val)
        oof_cat[val_idx] = cat_clf.predict_proba(X_val)
        m = score(y_cls_val, oof_cat[val_idx].argmax(axis=1))
        fold_scores_cat.append(m)
        print(f"  cat_mc:  F1={m['macro_f1']:.4f}  QWK={m['qwk']:.4f}  "
              f"acc={m['accuracy']:.4f}", flush=True)

        # ------ LightGBM regressor on resid, convert via frozen cutoffs
        reg = train_lightgbm_regressor(X_tr_c, y_res_tr, X_val_c, y_res_val)
        resid_pred = reg.predict(X_val_c)
        oof_ord_raw[val_idx] = resid_pred
        pred_cls = _classify(resid_pred, low, high)
        oof_ord[val_idx] = np.eye(3)[pred_cls]
        m = score(y_cls_val, pred_cls)
        fold_scores_ord.append(m)
        print(f"  lgb_reg: F1={m['macro_f1']:.4f}  QWK={m['qwk']:.4f}  "
              f"acc={m['accuracy']:.4f}", flush=True)

    oof_lgb_s = score(y_cls, oof_lgb.argmax(axis=1))
    oof_cat_s = score(y_cls, oof_cat.argmax(axis=1))
    oof_ord_s = score(y_cls, _classify(oof_ord_raw, low, high))
    print(f"\n=== OOF overall ===")
    print(f"  lgb_mc:  F1={oof_lgb_s['macro_f1']:.4f}  QWK={oof_lgb_s['qwk']:.4f}  "
          f"acc={oof_lgb_s['accuracy']:.4f}")
    print(f"  cat_mc:  F1={oof_cat_s['macro_f1']:.4f}  QWK={oof_cat_s['qwk']:.4f}  "
          f"acc={oof_cat_s['accuracy']:.4f}")
    print(f"  lgb_reg: F1={oof_ord_s['macro_f1']:.4f}  QWK={oof_ord_s['qwk']:.4f}  "
          f"acc={oof_ord_s['accuracy']:.4f}")

    # ------ Stacker: logistic regression on OOF probabilities
    print("\n=== Stacker (Logistic Regression on OOF probas) ===")
    stack_X = np.concatenate([oof_lgb, oof_cat, oof_ord], axis=1)
    scaler = StandardScaler()
    stack_Xs = scaler.fit_transform(stack_X)
    stacker = LogisticRegression(max_iter=1000, class_weight="balanced",
                                 random_state=SEED, C=1.0)
    # OOF stacker score via nested CV
    stack_pred = np.full(n_tr, -1, dtype=int)
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(stack_Xs, y_cls, groups), start=1):
        sc = LogisticRegression(max_iter=1000, class_weight="balanced",
                                random_state=SEED, C=1.0)
        sc.fit(stack_Xs[tr_idx], y_cls[tr_idx])
        stack_pred[val_idx] = sc.predict(stack_Xs[val_idx])
    stack_oof = score(y_cls, stack_pred)
    print(f"  stacker OOF: F1={stack_oof['macro_f1']:.4f}  "
          f"QWK={stack_oof['qwk']:.4f}  acc={stack_oof['accuracy']:.4f}")

    # ------ Refit all base models on full train, generate test probas
    print("\n=== Refit all base models on full train ===")
    uniq = train["author"].drop_duplicates().tolist()
    val_auth = set(uniq[int(len(uniq) * 0.85):])
    val_mask = train["author"].isin(val_auth).values

    X_fit_c, X_val_c = _lgbm_cat(X[~val_mask], X[val_mask])
    _, X_te_c = _lgbm_cat(X_fit_c, X_te)

    lgb_full = train_lightgbm(X_fit_c, y_cls[~val_mask], X_val_c, y_cls[val_mask])
    cat_full = train_catboost(X[~val_mask], y_cls[~val_mask],
                              X[val_mask], y_cls[val_mask])
    reg_full = train_lightgbm_regressor(X_fit_c, y_res[~val_mask],
                                         X_val_c, y_res[val_mask])

    test_lgb = lgb_full.predict_proba(X_te_c)
    test_cat = cat_full.predict_proba(X_te)
    test_reg_raw = reg_full.predict(X_te_c)
    test_reg_cls = _classify(test_reg_raw, low, high)
    test_ord = np.eye(3)[test_reg_cls]

    y_te = test["label"].values.astype(int)
    m_lgb = score(y_te, test_lgb.argmax(axis=1))
    m_cat = score(y_te, test_cat.argmax(axis=1))
    m_ord = score(y_te, test_reg_cls)
    print(f"  lgb_mc  test: F1={m_lgb['macro_f1']:.4f}  QWK={m_lgb['qwk']:.4f}  "
          f"acc={m_lgb['accuracy']:.4f}")
    print(f"  cat_mc  test: F1={m_cat['macro_f1']:.4f}  QWK={m_cat['qwk']:.4f}  "
          f"acc={m_cat['accuracy']:.4f}")
    print(f"  lgb_reg test: F1={m_ord['macro_f1']:.4f}  QWK={m_ord['qwk']:.4f}  "
          f"acc={m_ord['accuracy']:.4f}")

    # Final stacker trained on all OOF probas, applied to test
    stacker_full = LogisticRegression(max_iter=1000, class_weight="balanced",
                                      random_state=SEED, C=1.0)
    stacker_full.fit(scaler.fit_transform(stack_X), y_cls)
    test_stack_X = np.concatenate([test_lgb, test_cat, test_ord], axis=1)
    test_stack = stacker_full.predict(scaler.transform(test_stack_X))
    m_stack = score(y_te, test_stack)
    print(f"  STACK   test: F1={m_stack['macro_f1']:.4f}  QWK={m_stack['qwk']:.4f}  "
          f"acc={m_stack['accuracy']:.4f}")

    # Save predictions
    for name, pred in [("tier4_lgb", test_lgb.argmax(axis=1)),
                        ("tier4_cat", test_cat.argmax(axis=1)),
                        ("tier4_ordinal", test_reg_cls),
                        ("tier4_stack", test_stack)]:
        out = test[["author", "post"]].assign(label=pred.astype(int))
        p = RESULTS / "predictions" / f"{name}.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(p, index=False)
    print(f"  wrote tier4_{{lgb,cat,ordinal,stack}}.csv")

    (RESULTS / "tier4_cv.json").write_text(json.dumps({
        "seed": SEED, "n_splits": N_SPLITS,
        "oof": {"lgb_mc": oof_lgb_s, "cat_mc": oof_cat_s,
                "lgb_reg": oof_ord_s, "stacker": stack_oof},
        "test": {"lgb_mc": m_lgb, "cat_mc": m_cat,
                 "lgb_reg": m_ord, "stacker": m_stack},
    }, indent=2), encoding="utf-8")
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
