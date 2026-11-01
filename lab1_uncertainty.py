from __future__ import annotations

import numpy as np


def author_bootstrap(authors, truth, predictions, seed=42, samples=2000):
    unique, indices = np.unique(np.asarray(authors, dtype=str), return_inverse=True)
    draws = np.random.default_rng(seed).multinomial(len(unique), np.full(len(unique), 1 / len(unique)),
                                                  size=samples)
    scores = {}
    for name, predicted in predictions.items():
        matrix = np.zeros((len(unique), 3, 3), dtype=np.int64)
        np.add.at(matrix, (indices, np.asarray(truth, dtype=int), np.asarray(predicted, dtype=int)), 1)
        boot = np.einsum("ba,aij->bij", draws, matrix)
        numerator = 2 * np.diagonal(boot, axis1=1, axis2=2)
        denominator = boot.sum(axis=1) + boot.sum(axis=2)
        f1 = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float),
                       where=denominator != 0).mean(axis=1)
        scores[name] = f1
    interval = lambda values: [round(float(x), 4) for x in np.quantile(values, [.025, .975])]
    return {"method": "paired percentile bootstrap by author", "samples": samples,
            "seed": seed, "authors": len(unique),
            "macro_f1_95ci": {name: interval(values) for name, values in scores.items()},
            "selected_minus_baseline_95ci": interval(scores["selected_blend"] - scores["official_baseline"])}
