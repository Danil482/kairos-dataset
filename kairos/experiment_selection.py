"""Pure, deterministic stage-two search and acceptance rules; no data loading."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import f1_score


def parameter_grid() -> dict:
    """The twelve predeclared HGB recipes, as independent complete configs."""
    base = {"max_iter": 300, "learning_rate": 0.08, "max_leaf_nodes": 31,
            "min_samples_leaf": 20, "l2_regularization": 0.0,
            "class_weight": "balanced", "early_stopping": "auto"}
    overrides = {
        "base": {}, "leaves15": {"max_leaf_nodes": 15},
        "leaves63": {"max_leaf_nodes": 63}, "min50": {"min_samples_leaf": 50},
        "min100": {"min_samples_leaf": 100}, "l2_1": {"l2_regularization": 1.0},
        "l2_10": {"l2_regularization": 10.0}, "unweighted": {"class_weight": None},
        "slow": {"learning_rate": 0.04, "max_iter": 600},
        "combo15": {"max_leaf_nodes": 15, "min_samples_leaf": 50, "l2_regularization": 1.0},
        "combo31": {"max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 10.0},
        "combo63": {"max_leaf_nodes": 63, "min_samples_leaf": 100, "l2_regularization": 1.0},
    }
    return {name: {**base, **changes} for name, changes in overrides.items()}


def simplex_weights(families=("A", "B", "C"), step=0.25) -> list[dict]:
    """Enumerate simplex weights; standalones first, then fewer active families.

    Every dictionary includes every family, including zero weights. The step
    must divide one exactly to within floating-point roundoff. Within equal
    complexity, input family order determines the stable enumeration order.
    """
    families = tuple(families)
    if not families or len(set(families)) != len(families):
        raise ValueError("families must be nonempty and unique")
    if not math.isfinite(step) or not 0 < step <= 1:
        raise ValueError("step must be finite and in (0, 1]")
    units = round(1 / step)
    if not math.isclose(units * step, 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("step must divide one")

    def compositions(total, width):
        if width == 1:
            yield (total,)
        else:
            for first in range(total, -1, -1):
                for rest in compositions(total - first, width - 1):
                    yield (first, *rest)

    choices = sorted(compositions(units, len(families)),
                     key=lambda values: (sum(value > 0 for value in values),
                                         tuple(-value for value in values)))
    return [dict(zip(families, (value / units for value in values))) for values in choices]


def _probabilities(value, rows=None):
    result = np.asarray(value, dtype=np.float64)
    if (result.ndim != 2 or result.shape[1] != 3
            or (rows is not None and result.shape[0] != rows)):
        raise ValueError("probabilities must have shape (n_rows, 3)")
    if (not np.isfinite(result).all() or (result < 0).any() or (result > 1).any()
            or not np.allclose(result.sum(axis=1), 1.0, rtol=0, atol=1e-6)):
        raise ValueError("probabilities must be finite, in [0, 1], and sum to one")
    return result


def blend_arrays(mapping, weights) -> np.ndarray:
    """Linearly blend aligned [below, typical, above] probabilities.

    Row/key alignment belongs to the caller; this function checks shapes and
    distributions, and never changes inputs or renormalises the result.
    """
    if not weights or any(name not in mapping for name in weights):
        raise ValueError("weights require known probability families")
    values = np.asarray(list(weights.values()), dtype=np.float64)
    if (not np.isfinite(values).all() or (values < 0).any()
            or not np.isclose(values.sum(), 1.0, rtol=0, atol=1e-12)):
        raise ValueError("weights must be nonnegative, finite and sum to one")
    result = None
    for family, weight in weights.items():
        if weight == 0:
            continue
        probabilities = _probabilities(mapping[family], None if result is None else len(result))
        if result is None:
            result = np.zeros_like(probabilities)
        result += float(weight) * probabilities
    return result


def passes_gate(y, candidate_probs, reference_probs, fold_ids) -> dict:
    """Apply the fixed OOF gate, without selecting or fitting any model.

    Require pooled macro-F1 delta >= .002, positive deltas in at least two
    folds, and every fold delta >= -.005. Thresholds are inclusive; a 1e-12
    roundoff allowance prevents decimal boundary differences being rejected.
    Fold wins are strictly positive beyond that same numerical allowance.
    Scores use all three classes and zero_division=0, as the main experiment.
    """
    y, folds = np.asarray(y), np.asarray(fold_ids)
    if y.ndim != 1 or not len(y) or not np.isin(y, [0, 1, 2]).all():
        raise ValueError("y must be a nonempty vector of labels 0, 1, 2")
    if folds.ndim != 1 or len(folds) != len(y):
        raise ValueError("fold_ids must align with y")
    if any(value is None or (isinstance(value, (float, np.floating))
                            and not np.isfinite(value)) for value in folds):
        raise ValueError("fold_ids must not be missing")
    candidate = _probabilities(candidate_probs, len(y)).argmax(axis=1)
    reference = _probabilities(reference_probs, len(y)).argmax(axis=1)

    def delta(mask):
        return float(f1_score(y[mask], candidate[mask], labels=[0, 1, 2],
                              average="macro", zero_division=0)
                     - f1_score(y[mask], reference[mask], labels=[0, 1, 2],
                                average="macro", zero_division=0))

    pooled_delta = delta(np.ones(len(y), dtype=bool))
    fold_deltas = {str(fold): delta(folds == fold) for fold in np.unique(folds)}
    tolerance = 1e-12
    accepted = (pooled_delta >= 0.002 - tolerance
                and sum(value > tolerance for value in fold_deltas.values()) >= 2
                and min(fold_deltas.values()) >= -0.005 - tolerance)
    return {"accepted": bool(accepted), "pooled_delta": pooled_delta,
            "fold_deltas": fold_deltas}
