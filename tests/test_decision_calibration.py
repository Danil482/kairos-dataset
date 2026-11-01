"""Decision-rule integrity and author-isolated cross-calibration contracts."""
from __future__ import annotations

import unittest
import warnings
from unittest.mock import patch

import numpy as np

import calibrate_decision as calibration


def fixture():
    authors = np.repeat([f"fold{f}_author{a}" for f in range(3) for a in range(6)], 3)
    folds = np.repeat(np.arange(3), 18)
    truth = np.tile([0, 1, 2], 18)
    probabilities = np.tile([.34, .33, .33], (len(truth), 1))
    return truth, probabilities, folds, authors


class DecisionCalibrationTests(unittest.TestCase):
    def test_decide_preserves_inputs_and_common_scale_does_not_change_labels(self):
        probabilities = np.array([[.45, .50, .05], [.20, .39, .41]])
        weights = np.array([1.2, 1., .8])
        before_p, before_w = probabilities.copy(), weights.copy()
        np.testing.assert_array_equal(calibration.decide(probabilities, weights), [0, 1])
        np.testing.assert_array_equal(calibration.decide(probabilities, 7 * weights), [0, 1])
        np.testing.assert_array_equal(probabilities, before_p)
        np.testing.assert_array_equal(weights, before_w)
        np.testing.assert_array_equal(calibration.decide(probabilities, np.ones(3)), probabilities.argmax(1))

    def test_decide_rejects_invalid_probability_distributions(self):
        for probabilities in ([.2, .3, .5], [[.5, .5]], [[.2, .3, .6]],
                              [[np.nan, .5, .5]], [[np.inf, 0, 0]], [[-.1, .6, .5]]):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                calibration.decide(probabilities, [1, 1, 1])

    def test_decide_rejects_nonpositive_nonfinite_or_wrong_shape_weights(self):
        for weights in ([0, 1, 1], [-1, 1, 1], [1, np.nan, 1], [1, np.inf, 1],
                        [1, 1], [[1, 1, 1]]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                calibration.decide([[.2, .3, .5]], weights)

    def test_grid_uses_only_calibration_mask_and_recovers_needed_multipliers(self):
        truth = np.array([0, 2, 1, 1])
        probabilities = np.array([[.34, .40, .26], [.26, .40, .34], [0, 1, 0], [0, 1, 0.]])
        mask = np.array([True, True, False, False])
        weights, score = calibration.best_weights(truth, probabilities, mask)
        np.testing.assert_allclose(weights, [1.2, 1., 1.2])
        self.assertAlmostEqual(score, 2 / 3)
        truth[~mask] = [2, 0]
        probabilities[~mask] = [[1, 0, 0], [0, 0, 1]]
        other_weights, other_score = calibration.best_weights(truth, probabilities, mask)
        np.testing.assert_array_equal(weights, other_weights)
        self.assertEqual(score, other_score)

    def test_grid_prefers_identity_when_scores_tie(self):
        weights, score = calibration.best_weights(np.arange(3), np.eye(3), np.ones(3, dtype=bool))
        np.testing.assert_array_equal(weights, np.ones(3))
        self.assertEqual(score, 1)

    def test_cross_calibration_masks_are_author_disjoint_and_confined_to_same_outer_fold(self):
        truth, probabilities, folds, authors = fixture()
        masks = []

        def fitted(y, p, mask):
            masks.append(mask.copy())
            return np.ones(3), 0.0

        with patch.object(calibration, "best_weights", side_effect=fitted):
            predicted, halves = calibration.cross_calibrate(truth, probabilities, folds, authors)
        self.assertEqual(len(masks), 6)
        self.assertEqual(len(halves), 6)
        np.testing.assert_array_equal(np.sum(masks, axis=0), np.ones(len(truth)))
        np.testing.assert_array_equal(predicted, probabilities.argmax(1))
        for mask, half in zip(masks, halves):
            calibration_authors = set(authors[mask])
            evaluation_authors = set(half["evaluation_authors"])
            self.assertEqual(calibration_authors, set(half["calibration_authors"]))
            self.assertTrue(calibration_authors.isdisjoint(evaluation_authors))
            self.assertEqual(set(folds[mask]), {half["fold"]})
            self.assertEqual(calibration_authors | evaluation_authors, set(authors[folds == half["fold"]]))
            for author in calibration_authors:
                self.assertTrue(mask[authors == author].all())

    def test_predictions_ignore_own_half_labels_and_other_outer_fold_labels(self):
        truth, probabilities, folds, authors = fixture()

        def label_sensitive_fit(y, p, mask):
            # Deliberately sensitive: leakage would flip class 0 to class 2.
            weights = [0.8, 1, 1.2] if np.mean(y[mask]) > 1.5 else [1.2, 1, .8]
            return np.array(weights), 0.0

        with warnings.catch_warnings(), patch.object(calibration, "best_weights", side_effect=label_sensitive_fit):
            warnings.simplefilter("ignore")
            predicted, halves = calibration.cross_calibrate(truth, probabilities, folds, authors)
            half = halves[0]
            evaluation = np.isin(authors, half["evaluation_authors"])
            calibration_rows = np.isin(authors, half["calibration_authors"])
            changed = truth.copy()
            changed[evaluation | (folds != half["fold"])] = 2
            other, _ = calibration.cross_calibrate(changed, probabilities, folds, authors)
        np.testing.assert_array_equal(predicted[evaluation], other[evaluation])
        # Positive control: these rows calibrate on the changed opposite half.
        self.assertTrue(np.all(predicted[calibration_rows] != other[calibration_rows]))

    def test_cross_calibration_author_halves_survive_row_permutation(self):
        truth, probabilities, folds, authors = fixture()
        order = np.random.default_rng(91).permutation(len(truth))
        predicted, halves = calibration.cross_calibrate(truth, probabilities, folds, authors)
        permuted, other_halves = calibration.cross_calibrate(truth[order], probabilities[order], folds[order], authors[order])
        np.testing.assert_array_equal(predicted[order], permuted)
        for first, second in zip(halves, other_halves):
            self.assertEqual(first["calibration_authors"], second["calibration_authors"])
            self.assertEqual(first["evaluation_authors"], second["evaluation_authors"])
            self.assertEqual(first["weights"], second["weights"])

    def test_author_crossing_outer_folds_is_rejected(self):
        truth, probabilities, folds, authors = fixture()
        authors[18] = authors[0]
        with self.assertRaisesRegex(ValueError, "author crosses"):
            calibration.cross_calibrate(truth, probabilities, folds, authors)


if __name__ == "__main__":
    unittest.main()
