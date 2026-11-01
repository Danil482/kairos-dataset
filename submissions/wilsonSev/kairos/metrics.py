"""The three numbers every result is reported with."""

from __future__ import annotations

from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


def score(y_true, y_pred) -> dict[str, float]:
    return {
        # The headline. It averages the three classes equally, so a model cannot
        # win it by answering "typical" for everything, as it can win accuracy.
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        # The classes are ordered: calling "above" a post that was "below" is a
        # worse error than calling it "typical". Quadratic kappa counts that;
        # macro-F1 and accuracy do not.
        "qwk": round(float(cohen_kappa_score(y_true, y_pred, weights="quadratic")), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
    }
