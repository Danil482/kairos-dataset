"""Paired author-cluster resampling against an explicit row-expanded reference."""

from __future__ import annotations

import unittest

import numpy as np
from sklearn.metrics import f1_score

from kairos.experiment_uncertainty import paired_author_bootstrap


def fixture():
    # Deliberately unequal author sizes and opposite strengths across authors.
    truth = np.array([0, 1, 2, 2, 2, 1, 0, 0, 1])
    candidate = np.array([0, 1, 0, 0, 0, 2, 2, 0, 1])
    reference = np.array([1, 0, 2, 2, 2, 1, 0, 1, 1])
    authors = np.array(["a", "a", "b", "b", "b", "b", "b", "c", "c"])
    return truth, candidate, reference, authors


def explicit_cluster_differences(truth, candidate, reference, authors, iterations, seed):
    unique = np.unique(authors)
    generator = np.random.default_rng(seed)
    differences = []
    for _ in range(iterations):
        sampled = generator.integers(0, len(unique), size=len(unique))
        rows = np.concatenate([np.flatnonzero(authors == unique[index]) for index in sampled])
        scores = [f1_score(truth[rows], predicted[rows], labels=[0, 1, 2],
                           average="macro", zero_division=0)
                  for predicted in (candidate, reference)]
        differences.append(scores[0] - scores[1])
    return np.asarray(differences)


class AuthorBootstrapTests(unittest.TestCase):
    def test_matches_explicit_whole_author_sampling_with_unequal_cluster_sizes(self):
        data = fixture()
        result = paired_author_bootstrap(*data, n_resamples=200, seed=8)
        expected = explicit_cluster_differences(*data, iterations=200, seed=8)
        low, high = np.quantile(expected, [0.025, 0.975])
        self.assertAlmostEqual(result["ci_low"], low)
        self.assertAlmostEqual(result["ci_high"], high)
        truth, candidate, reference, _ = data
        difference = f1_score(truth, candidate, labels=[0, 1, 2], average="macro", zero_division=0) - f1_score(
            truth, reference, labels=[0, 1, 2], average="macro", zero_division=0)
        self.assertAlmostEqual(result["observed_delta"], difference)
        self.assertEqual(result["n_authors"], 3)
        self.assertEqual(result["n_posts"], 9)

    def test_identical_models_have_exact_zero_paired_interval(self):
        truth, candidate, _, authors = fixture()
        result = paired_author_bootstrap(truth, candidate, candidate, authors, n_resamples=100)
        self.assertEqual(result["observed_delta"], 0)
        self.assertEqual(result["ci_low"], 0)
        self.assertEqual(result["ci_high"], 0)

    def test_seed_and_row_permutation_are_reproducible(self):
        data = fixture()
        first = paired_author_bootstrap(*data, n_resamples=75, seed=42)
        second = paired_author_bootstrap(*data, n_resamples=75, seed=42)
        self.assertEqual(first, second)
        permutation = np.array([4, 7, 0, 2, 8, 1, 3, 6, 5])
        reordered = paired_author_bootstrap(*(values[permutation] for values in data),
                                            n_resamples=75, seed=42)
        self.assertEqual(first, reordered)

    def test_swapping_models_negates_delta_and_interval(self):
        truth, candidate, reference, authors = fixture()
        forward = paired_author_bootstrap(truth, candidate, reference, authors, n_resamples=300)
        reverse = paired_author_bootstrap(truth, reference, candidate, authors, n_resamples=300)
        self.assertAlmostEqual(forward["observed_delta"], -reverse["observed_delta"])
        self.assertAlmostEqual(forward["ci_low"], -reverse["ci_high"])
        self.assertAlmostEqual(forward["ci_high"], -reverse["ci_low"])

    def test_missing_classes_still_count_as_zero_in_macro_average(self):
        result = paired_author_bootstrap([0, 0], [0, 0], [1, 1], ["only", "only"], n_resamples=20)
        self.assertAlmostEqual(result["candidate_macro_f1"], 1 / 3)
        self.assertEqual(result["reference_macro_f1"], 0)
        self.assertAlmostEqual(result["ci_low"], 1 / 3)
        self.assertAlmostEqual(result["ci_high"], 1 / 3)
        self.assertEqual(result["labels"], [0, 1, 2])
        self.assertEqual(result["zero_division"], 0)

    def test_invalid_alignment_labels_and_author_ids_fail(self):
        data = fixture()
        for replacement in ([0, 1], [3] * 9, np.zeros((9, 1))):
            with self.subTest(replacement=replacement):
                with self.assertRaises(ValueError):
                    paired_author_bootstrap(data[0], replacement, data[2], data[3])
        for authors in ([None] * 9, [""] * 9, ["author"]):
            with self.subTest(authors=authors):
                with self.assertRaises(ValueError):
                    paired_author_bootstrap(data[0], data[1], data[2], authors)

    def test_bad_sampling_configuration_fails(self):
        for config in ({"n_resamples": 0}, {"n_resamples": True}, {"seed": -1},
                       {"confidence": 1}, {"confidence": 0}):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    paired_author_bootstrap(*fixture(), **config)


if __name__ == "__main__":
    unittest.main()
