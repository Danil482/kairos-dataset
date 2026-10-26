"""Guard against author leakage and silently swapped class probabilities."""
import unittest
import numpy as np
import pandas as pd

from scripts.lecture_models import ordered_probabilities, split_by_plan


class LectureProtocolTests(unittest.TestCase):
    def setUp(self):
        self.table = pd.DataFrame({"author": ["A", "A", "B", "C"],
            "post": ["p1", "p2", "p3", "p4"], "split": ["train"] * 4})
        self.plan = dict(fitting_authors=["A"], calibration_authors=["B"], validation_authors=["C"])

    def test_full_authors_are_kept_together_and_overlap_is_rejected(self):
        fitting, calibration, validation = split_by_plan(self.table, self.plan)
        self.assertEqual(fitting.post.tolist(), ["p1", "p2"])
        self.assertEqual(calibration.author.tolist(), ["B"])
        self.assertEqual(validation.author.tolist(), ["C"])
        self.plan["validation_authors"].append("A")
        with self.assertRaisesRegex(ValueError, "overlap"):
            split_by_plan(self.table, self.plan)

    def test_official_test_or_missing_authors_cannot_enter_training(self):
        contaminated = self.table.copy()
        contaminated.loc[3, "split"] = "test"
        with self.assertRaisesRegex(ValueError, "official train"):
            split_by_plan(contaminated, self.plan)
        self.plan["validation_authors"] = []
        with self.assertRaisesRegex(ValueError, "cover"):
            split_by_plan(self.table, self.plan)

    def test_reversed_class_order_is_rejected_before_fusion(self):
        class ReversedModel:
            classes_ = np.array([2, 1, 0])
            def predict_proba(self, matrix):
                return np.array([[.8, .1, .1]])
        with self.assertRaisesRegex(ValueError, "order"):
            ordered_probabilities(ReversedModel(), None)


if __name__ == "__main__":
    unittest.main()
