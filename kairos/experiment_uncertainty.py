"""Paired uncertainty estimates for predictions on the same held-out authors."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _macro_f1(confusion: np.ndarray) -> np.ndarray:
    """Macro-F1 over the fixed labels 0/1/2, with zero_division=0."""
    diagonal = np.diagonal(confusion, axis1=-2, axis2=-1)
    denominator = confusion.sum(axis=-1) + confusion.sum(axis=-2)
    per_class = np.divide(2.0 * diagonal, denominator,
                          out=np.zeros_like(diagonal, dtype=float), where=denominator > 0)
    return per_class.mean(axis=-1)


def paired_author_bootstrap(y_true, candidate_predictions, reference_predictions, authors,
                            *, n_resamples: int = 1000, seed: int = 42,
                            confidence: float = 0.95) -> dict:
    """Percentile interval for candidate minus reference macro-F1.

    Inputs must be aligned, one row per post. Authors are sampled uniformly with
    replacement, drawing the original number of authors each time. Every draw
    retains all posts of the chosen author; both models use the same draws.
    Author confusion matrices avoid copying posts for each bootstrap replicate.

    The metric remains post-level macro-F1, not the average of author F1 scores.
    Classes absent from a replicate contribute zero, matching sklearn's
    f1_score(labels=[0, 1, 2], average="macro", zero_division=0).

    This interval describes author sampling uncertainty for these fixed paired
    predictions. It does not undo model-selection bias or estimate training
    randomness; for selected OOF models report that limitation alongside it.
    """
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, (int, np.integer)) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(confidence, bool) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")

    arrays = []
    for name, values in (("y_true", y_true), ("candidate_predictions", candidate_predictions),
                         ("reference_predictions", reference_predictions)):
        values = np.asarray(values)
        if values.ndim != 1 or not len(values) or not np.isin(values, [0, 1, 2]).all():
            raise ValueError(f"{name} must be a nonempty 1D sequence of labels 0, 1, 2")
        arrays.append(values.astype(np.int64))
    truth, candidate, reference = arrays
    author_ids = np.asarray(authors)
    if author_ids.ndim != 1 or any(len(values) != len(truth) for values in arrays[1:]) or len(author_ids) != len(truth):
        raise ValueError("Labels, predictions and author IDs must have the same length")
    if pd.isna(author_ids).any() or any(isinstance(value, str) and not value.strip() for value in author_ids):
        raise ValueError("Author IDs must not be missing or empty")
    # Canonical author order makes a row permutation irrelevant to the seeded draws.
    codes, unique_authors = pd.factorize(author_ids, sort=True)
    n_authors = len(unique_authors)
    author_confusion = np.zeros((n_authors, 2, 3, 3), dtype=np.int64)
    np.add.at(author_confusion, (codes, 0, truth, candidate), 1)
    np.add.at(author_confusion, (codes, 1, truth, reference), 1)
    observed = _macro_f1(author_confusion.sum(axis=0))

    flattened = author_confusion.reshape(n_authors, 18)
    generator = np.random.default_rng(seed)
    differences = np.empty(n_resamples, dtype=float)
    for iteration in range(n_resamples):
        sampled = generator.integers(0, n_authors, size=n_authors)
        multiplicities = np.bincount(sampled, minlength=n_authors)
        paired_confusion = (multiplicities @ flattened).reshape(2, 3, 3)
        scores = _macro_f1(paired_confusion)
        differences[iteration] = scores[0] - scores[1]
    tail = (1 - confidence) / 2
    low, high = np.quantile(differences, [tail, 1 - tail], method="linear")
    return {
        "observed_delta": float(observed[0] - observed[1]),
        "candidate_macro_f1": float(observed[0]),
        "reference_macro_f1": float(observed[1]),
        "ci_low": float(low), "ci_high": float(high),
        "confidence": float(confidence), "iterations": int(n_resamples), "seed": int(seed),
        "n_authors": n_authors, "n_posts": len(truth),
        "method": "paired_author_cluster_percentile_bootstrap",
        "sampling_unit": "author", "paired": True,
        "labels": [0, 1, 2], "zero_division": 0,
        "interval_scope": "author sampling of fixed predictions; excludes training and model-selection uncertainty",
    }
