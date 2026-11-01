"""Fixed search candidates and acceptance boundaries, without training."""

import unittest

import numpy as np

from kairos.experiment_selection import (blend_arrays, parameter_grid, passes_gate,
                                         simplex_weights)


def gate_fixture(correct_per_fold):
    # Balanced cyclic errors make every class's F1 exactly correct / 1000.
    # Thus pooled F1 is the mean of fold F1, allowing real boundary examples.
    y_fold = np.repeat(np.arange(3), 1000)

    def prediction(correct):
        return np.concatenate([np.r_[np.full(correct, label),
                                     np.full(1000 - correct, (label + 1) % 3)]
                               for label in range(3)])

    y = np.tile(y_fold, len(correct_per_fold))
    reference = np.tile(prediction(600), len(correct_per_fold))
    candidate = np.concatenate([prediction(correct) for correct in correct_per_fold])
    folds = np.repeat(np.arange(len(correct_per_fold)), len(y_fold))
    return y, np.eye(3)[candidate], np.eye(3)[reference], folds


class SelectionUtilityTest(unittest.TestCase):
    def test_parameter_grid_is_complete_fixed_and_independent(self):
        grid = parameter_grid()
        self.assertEqual(list(grid), ["base", "leaves15", "leaves63", "min50", "min100",
                                      "l2_1", "l2_10", "unweighted", "slow",
                                      "combo15", "combo31", "combo63"])
        self.assertEqual(grid["base"], {"max_iter": 300, "learning_rate": 0.08,
                                       "max_leaf_nodes": 31, "min_samples_leaf": 20,
                                       "l2_regularization": 0, "class_weight": "balanced",
                                       "early_stopping": "auto"})
        self.assertTrue(all(set(config) == set(grid["base"]) for config in grid.values()))
        self.assertEqual(grid["slow"]["max_iter"], 600)
        self.assertEqual(grid["slow"]["learning_rate"], 0.04)
        self.assertIsNone(grid["unweighted"]["class_weight"])
        self.assertTrue(all(config["class_weight"] == "balanced"
                            for name, config in grid.items() if name != "unweighted"))
        for name, expected in [("combo15", (15, 50, 1)), ("combo31", (31, 50, 10)),
                               ("combo63", (63, 100, 1))]:
            self.assertEqual(tuple(grid[name][key] for key in
                                   ["max_leaf_nodes", "min_samples_leaf", "l2_regularization"]), expected)
        grid["base"]["max_iter"] = 7
        self.assertEqual(grid["leaves15"]["max_iter"], 300)
        self.assertEqual(parameter_grid()["base"]["max_iter"], 300)

    def test_quarter_simplex_has_all_fifteen_points_including_standalones(self):
        weights = simplex_weights()
        self.assertEqual(len(weights), 15)
        self.assertEqual(len({tuple(item.values()) for item in weights}), 15)
        for item in weights:
            self.assertEqual(set(item), {"A", "B", "C"})
            self.assertEqual(sum(item.values()), 1)
            self.assertTrue(all(value >= 0 and value * 4 == int(value * 4) for value in item.values()))
        self.assertEqual(weights[:3], [{"A": 1, "B": 0, "C": 0},
                                      {"A": 0, "B": 1, "C": 0},
                                      {"A": 0, "B": 0, "C": 1}])
        self.assertIn({"A": 0.25, "B": 0.5, "C": 0.25}, weights)
        self.assertEqual(len(simplex_weights(("X", "Y"), step=0.5)), 3)

    def test_invalid_simplex_requests_fail(self):
        for families, step in [((), 0.25), (("A", "A"), 0.25), (("A",), 0),
                               (("A",), 0.3), (("A",), float("nan"))]:
            with self.subTest(families=families, step=step), self.assertRaises(ValueError):
                simplex_weights(families, step)

    def test_blend_is_exact_linear_probability_combination_without_input_changes(self):
        a = np.array([[1, 0, 0], [0.2, 0.3, 0.5]], dtype=float)
        b = np.array([[0, 0, 1], [0.4, 0.4, 0.2]], dtype=float)
        original_a, original_b = a.copy(), b.copy()
        result = blend_arrays({"A": a, "B": b}, {"A": 0.25, "B": 0.75})
        np.testing.assert_allclose(result, [[0.25, 0, 0.75], [0.35, 0.375, 0.275]])
        np.testing.assert_allclose(result.sum(axis=1), 1)
        np.testing.assert_array_equal(a, original_a)
        np.testing.assert_array_equal(b, original_b)
        np.testing.assert_array_equal(blend_arrays({"A": a}, {"A": 1}), a)

    def test_blend_rejects_invalid_weights_shapes_and_probabilities(self):
        good = np.array([[1, 0, 0]], dtype=float)
        for weights in [{}, {"X": 1}, {"A": -1, "B": 2}, {"A": 0.8}, {"A": np.nan}]:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                blend_arrays({"A": good, "B": good}, weights)
        for bad in [np.ones((2, 3)) / 3, [[np.nan, 0, 1]], [[0, 0, 0]], [[-1, 1, 1]], [[1, 0]]]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                blend_arrays({"A": good, "B": bad}, {"A": 0.5, "B": 0.5})

    def test_gate_accepts_exact_pooled_and_worst_fold_boundaries(self):
        result = passes_gate(*gate_fixture([610, 601, 595]))
        self.assertTrue(result["accepted"])
        self.assertAlmostEqual(result["pooled_delta"], 0.002)
        self.assertAlmostEqual(result["fold_deltas"]["2"], -0.005)
        self.assertEqual(set(result["fold_deltas"]), {"0", "1", "2"})

    def test_gate_rejects_pooled_delta_below_boundary(self):
        result = passes_gate(*gate_fixture([609, 601, 595]))
        self.assertFalse(result["accepted"])
        self.assertLess(result["pooled_delta"], 0.002)

    def test_gate_rejects_one_bad_fold_despite_good_pooled_score(self):
        result = passes_gate(*gate_fixture([620, 610, 594]))
        self.assertFalse(result["accepted"])
        self.assertGreater(result["pooled_delta"], 0.002)
        self.assertLess(result["fold_deltas"]["2"], -0.005)

    def test_gate_requires_two_strict_wins_not_ties(self):
        result = passes_gate(*gate_fixture([606, 600, 600]))
        self.assertAlmostEqual(result["pooled_delta"], 0.002)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["fold_deltas"]["1"], 0)

    def test_gate_checks_alignment_and_valid_labels(self):
        y, candidate, reference, folds = gate_fixture([610, 601, 595])
        with self.assertRaisesRegex(ValueError, "align"):
            passes_gate(y, candidate, reference, folds[:-1])
        with self.assertRaisesRegex(ValueError, "labels"):
            passes_gate(y + 3, candidate, reference, folds)
        with self.assertRaisesRegex(ValueError, "shape"):
            passes_gate(y, candidate[:-1], reference, folds)


if __name__ == "__main__":
    unittest.main()
