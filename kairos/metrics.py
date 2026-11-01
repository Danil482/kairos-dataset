from __future__ import annotations
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

def score(y_true, y_pred) -> dict[str, float]:
    return {'macro_f1': round(float(f1_score(y_true, y_pred, average='macro')), 4), 'qwk': round(float(cohen_kappa_score(y_true, y_pred, weights='quadratic')), 4), 'accuracy': round(float(accuracy_score(y_true, y_pred)), 4)}
