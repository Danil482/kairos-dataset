"""Quality metrics for the three post classes."""

from __future__ import annotations

from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


def score(y_true, y_pred) -> dict[str, float]:
    f1_by_class = f1_score(
        y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
    )
    return {
        # The headline. It averages the three classes equally, so a model cannot
        # win it by answering "typical" for everything, as it can win accuracy.
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "f1_below": round(float(f1_by_class[0]), 4),
        "f1_typical": round(float(f1_by_class[1]), 4),
        "f1_above": round(float(f1_by_class[2]), 4),
        # The classes are ordered: calling "above" a post that was "below" is a
        # worse error than calling it "typical". Quadratic kappa counts that;
        # macro-F1 and accuracy do not.
        "qwk": round(float(cohen_kappa_score(y_true, y_pred, weights="quadratic")), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
    }
