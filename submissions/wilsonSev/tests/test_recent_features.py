import unittest
import numpy as np
import pandas as pd
from kairos.recent_features import recent_author_features, RECENT_COLUMNS


class RecentTests(unittest.TestCase):
    def test_exact_history_ties_and_no_future_or_current_leak(self):
        p = pd.DataFrame(
            {
                "author": ["A"] * 6 + ["B"],
                "post": ["0", "1", "2", "3", "4", "missing", "b"],
                "posted_at": pd.to_datetime(
                    [
                        "2020-01-01",
                        "2020-01-02",
                        "2020-01-02",
                        "2020-01-03",
                        "2020-01-04",
                        None,
                        "2020-01-01",
                    ]
                ),
                "reactions": [1, 9, 99, 999, 20, 999999, 999999],
            }
        )
        got = recent_author_features(p)
        np.testing.assert_allclose(got.loc[2, RECENT_COLUMNS].to_numpy(dtype=float), 0)
        hist = np.log1p([1, 9, 99, 999])
        usual = np.median(hist)
        np.testing.assert_allclose(
            got.loc[4, RECENT_COLUMNS].to_numpy(dtype=float),
            [
                hist[-3:].mean() - usual,
                hist.mean() - usual,
                hist[-1] - usual,
                hist[-3:].std(),
            ],
        )
        np.testing.assert_allclose(got.loc[5, RECENT_COLUMNS].to_numpy(dtype=float), 0)
        changed = p.copy()
        changed.loc[3:, "reactions"] = 1
        pd.testing.assert_frame_equal(
            got.iloc[:4], recent_author_features(changed).iloc[:4]
        )
        prefix = recent_author_features(p.iloc[:4])
        pd.testing.assert_frame_equal(got.iloc[:4], prefix)
        shuffled = (
            recent_author_features(p.sample(frac=1, random_state=42))
            .sort_values(["author", "post"])
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            got.sort_values(["author", "post"]).reset_index(drop=True), shuffled
        )
