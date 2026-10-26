import unittest

import numpy as np
import pandas as pd

from student_features import build_features, normalize_text


class HistoryTests(unittest.TestCase):
    def sample(self):
        stamps = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-10", "2024-02-10", "2024-05-01"], utc=True)
        return pd.DataFrame(dict(author=["a"] * 6, post=list("abcdef"), posted_at=stamps,
                                 crawled_at=pd.Timestamp("2025-01-01", tz="UTC"),
                                 reactions=[0, 2, 4, 8, 16, 32], text=["hello #test"] * 6,
                                 content_type=["text"] * 6, own=True))

    def test_current_and_future_reactions_cannot_change_features(self):
        posts = self.sample()
        original, numeric = build_features(posts)
        for i in range(len(posts)):
            modified = posts.copy()
            modified.loc[i:, "reactions"] = 99999
            actual, _ = build_features(modified)
            pd.testing.assert_series_equal(original.loc[i, numeric], actual.loc[i, numeric])

    def test_equal_time_is_not_history(self):
        result, _ = build_features(self.sample())
        self.assertEqual(result.loc[1, "hist_n"], 1)
        self.assertEqual(result.loc[2, "hist_n"], 1)

    def test_windows_and_datetime_units(self):
        posts = self.sample()
        result, numeric = build_features(posts)
        self.assertEqual(result.loc[3, "posts_7d"], 0)
        self.assertEqual(result.loc[3, "posts_30d"], 3)
        self.assertEqual(result.loc[3, "days_since_prev"], 8)
        for unit in ["ms", "us", "ns"]:
            variant = posts.copy()
            variant["posted_at"] = variant.posted_at.dt.as_unit(unit)
            actual, _ = build_features(variant)
            pd.testing.assert_frame_equal(result[numeric], actual[numeric])

    def test_author_isolation(self):
        first = self.sample()
        second = first.assign(author="b", reactions=5000)
        expected, numeric = build_features(first)
        mixed, _ = build_features(pd.concat([first, second], ignore_index=True))
        pd.testing.assert_frame_equal(expected[numeric], mixed.loc[mixed.author == "a", numeric])

    def test_explicit_feature_whitelist(self):
        result, numeric = build_features(self.sample().assign(label=1, resid=100))
        self.assertFalse({"label", "resid", "reactions", "log_reactions"} & set(result.columns))
        self.assertFalse({"age_days", "log_followers"} & set(numeric))
        self.assertEqual(normalize_text("⟨AUTHOR_123⟩ ⟨PERSON_44⟩"), "⟨AUTHOR⟩ ⟨PERSON⟩")


if __name__ == "__main__":
    unittest.main()
