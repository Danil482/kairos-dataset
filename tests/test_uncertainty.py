from __future__ import annotations

import unittest

import numpy as np

from lab1_uncertainty import author_bootstrap


class BootstrapTests(unittest.TestCase):
    def test_paired_identical_predictions_have_zero_difference(self):
        truth = np.tile([0, 1, 2], 8)
        authors = np.repeat(["A", "B", "C", "D"], 6)
        result = author_bootstrap(authors, truth, {
            "official_baseline": truth, "selected_blend": truth}, samples=100)
        self.assertEqual(result["macro_f1_95ci"]["selected_blend"], [1.0, 1.0])
        self.assertEqual(result["selected_minus_baseline_95ci"], [0.0, 0.0])
        self.assertEqual(result["authors"], 4)


if __name__ == "__main__":
    unittest.main()
