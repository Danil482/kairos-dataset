from __future__ import annotations

import unittest

import pandas as pd

from kairos.label import author_level
from lab1_features import CATEGORICAL, build, clean_text


def posts():
    return pd.DataFrame({
        "author": ["AUTHOR_1"] * 12 + ["AUTHOR_2"] * 12,
        "post": [f"POST_{i}" for i in range(24)],
        "posted_at": list(pd.date_range("2025-01-01", periods=12, tz="UTC")) * 2,
        "crawled_at": pd.Timestamp("2026-01-01", tz="UTC"),
        "reactions": [2, 4, 8, 16, 4, 0, 18, 7, 3, 1, 20, 6] * 2,
        "text": ["A useful post?\n#data ⟨ORG_42⟩"] * 24,
        "content_type": ["text"] * 24, "own": True,
    })


class HistoryTests(unittest.TestCase):
    def matrix(self, raw):
        df, numeric = build(author_level(raw), pd.Series({"AUTHOR_1": 100, "AUTHOR_2": 200}))
        return df.set_index("post")[numeric + CATEGORICAL], numeric

    def test_current_engagement_does_not_change_current_features(self):
        raw = posts()
        before, _ = self.matrix(raw)
        raw.loc[8, "reactions"] = 1_000_000
        after, _ = self.matrix(raw)
        pd.testing.assert_frame_equal(before.loc[[f"POST_{i}" for i in range(9)]],
                                      after.loc[[f"POST_{i}" for i in range(9)]])
        self.assertNotEqual(before.loc["POST_9", "log_reactions_lag1"],
                            after.loc["POST_9", "log_reactions_lag1"])

    def test_future_outcomes_do_not_change_past_features(self):
        raw = posts()
        before, _ = self.matrix(raw)
        raw.loc[10:11, "reactions"] = [12345, 98765]
        after, _ = self.matrix(raw)
        pd.testing.assert_frame_equal(before.loc[[f"POST_{i}" for i in range(10)]],
                                      after.loc[[f"POST_{i}" for i in range(10)]])

    def test_history_stays_inside_author(self):
        raw = posts()
        before, numeric = self.matrix(raw)
        raw.loc[:11, "reactions"] *= 1000
        after, _ = self.matrix(raw)
        pd.testing.assert_frame_equal(before.loc[[f"POST_{i}" for i in range(12, 24)]],
                                      after.loc[[f"POST_{i}" for i in range(12, 24)]])
        self.assertFalse(set(numeric) & {"reactions", "log_reactions", "resid", "label", "author", "post"})

    def test_placeholder_identity_removed(self):
        self.assertEqual(clean_text("⟨PERSON_123⟩ at ⟨ORG_42⟩"), "PERSON  at  ORG")
        self.assertEqual(clean_text(""), "EMPTY_POST")
        self.assertNotIn("ORG_123", clean_text("cut placeholder ⟨ORG_123 here"))
        self.assertEqual(clean_text("⟨POST_77⟩"), "POST")


if __name__ == "__main__":
    unittest.main()
