import numpy as np
from sklearn.metrics import f1_score
import lightgbm as lgb

from .config import DENSE_PARAMS, TEXT_PARAMS, BLEND_DENSE, BLEND_TEXT


def fit_dense(X_train, y_train, X_eval, seeds=(42, 7, 123)):
    probs = []
    for s in seeds:
        params = dict(DENSE_PARAMS, random_state=s)
        m = lgb.LGBMClassifier(**params)
        m.fit(X_train, y_train)
        probs.append(m.predict_proba(X_eval))
    return np.mean(probs, axis=0)

def fit_text(X_train, y_train, X_eval, seeds=(42, 67, 58)):
    probs = []
    for s in seeds:
        params = dict(TEXT_PARAMS, random_state=s)
        m = lgb.LGBMClassifier(**params)
        m.fit(X_train, y_train)
        probs.append(m.predict_proba(X_eval))
    return np.mean(probs, axis=0)

def blend(p_dense, p_text):
    return BLEND_DENSE * p_dense + BLEND_TEXT * p_text

def fit_catboost(X_train, y_train, X_eval):
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(
        iterations=800,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiClass",
        classes_count=3,
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_eval)


def calibrate_weights(proba, y_true, n_grid=15):
    best_w = np.array([1.0, 1.0, 1.0])
    best_f1 = f1_score(y_true, proba.argmax(axis=1), average="macro")
    grid = np.linspace(0.5, 2.5, n_grid)
    for w0 in grid:
        for w2 in grid:
            w = np.array([w0, 1.0, w2])
            pred = (proba * w).argmax(axis=1)
            f1 = f1_score(y_true, pred, average="macro")
            if f1 > best_f1:
                best_f1 = f1
                best_w = w

    return best_w, best_f1