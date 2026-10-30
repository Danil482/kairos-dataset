"""BM25 history features: class separation, score formula, and temporal boundaries."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from kairos import features


class BM25FeaturesTests(unittest.TestCase):
    def setUp(self):
        for name, value in {
            "N_LAST": 20,
            "BM25_TOP_N": 12,
            "BM25_K1": 1.5,
            "BM25_B": 0.75,
        }.items():
            setting = patch.object(features, name, value)
            setting.start()
            self.addCleanup(setting.stop)
        # All seven training texts are 'alpha': df = 7, avgdl = 1.
        self.score = np.log(16 / 15)

    @staticmethod
    def history(past_residuals, past_texts=None, query="alpha"):
        calibration = pd.DataFrame(
            {
                "author": ["calibration"] * 7,
                "post": [f"train_{i}" for i in range(7)],
                "posted_at": pd.date_range("2024-01-01", periods=7, tz="UTC"),
                "text": ["alpha"] * 7,
                # Train cutoffs are -0.8 and 0.8.
                "resid": [-2, -1, 0, 0, 0, 1, 2],
            }
        )
        n_posts = len(past_residuals) + 1
        posts = pd.DataFrame(
            {
                "author": ["target"] * n_posts,
                "post": [f"post_{i}" for i in range(n_posts)],
                "posted_at": pd.date_range("2025-01-01", periods=n_posts, tz="UTC"),
                "text": (past_texts or ["alpha"] * len(past_residuals)) + [query],
                "resid": list(past_residuals) + [999],
            }
        )
        return pd.concat([calibration, posts], ignore_index=True), np.arange(7)

    def test_equal_total_residuals_keep_different_class_distributions(self):
        typical_history, train_rows = self.history([0, 0, 0, 0, 0])
        mixed_history, _ = self.history([-1, -1, -1, 1.5, 1.5])
        typical = features._text_bm25(typical_history, train_rows)[-1]
        mixed = features._text_bm25(mixed_history, train_rows)[-1]
        np.testing.assert_allclose(
            typical, [0, 0, 0, 5, 5 * self.score, 0, 0, 0, 0], atol=1e-6
        )
        np.testing.assert_allclose(
            mixed,
            [3, 3 * self.score, -3 * self.score, 0, 0, 0, 2, 2 * self.score, 3 * self.score],
            atol=1e-6,
        )
        self.assertEqual(typical[::3].sum(), 5)
        self.assertEqual(mixed[::3].sum(), 5)
        self.assertAlmostEqual(float(typical[2::3].sum()), float(mixed[2::3].sum()))

    def test_term_frequency_and_full_document_length_follow_bm25(self):
        history, train_rows = self.history(
            [0, 0, 0], ["alpha", "alpha alpha", "alpha unseenword"]
        )
        actual = features._text_bm25(history, train_rows)[-1]
        # With k1=1.5, b=0.75, avgdl=1, the three TF factors are:
        # 2.5 / 2.5, 5 / 4.625, 2.5 / 3.625.
        expected_sum = self.score * (1 + 40 / 37 + 20 / 29)
        np.testing.assert_allclose(
            actual, [0, 0, 0, 3, expected_sum, 0, 0, 0, 0], atol=1e-6
        )
        repeated_query = history.copy()
        repeated_query.loc[repeated_query.index[-1], "text"] = "alpha alpha"
        np.testing.assert_allclose(
            features._text_bm25(repeated_query, train_rows)[-1], actual, atol=1e-6
        )

    def test_only_top_twelve_positive_matches_are_counted(self):
        history, train_rows = self.history([0] * 15)
        actual = features._text_bm25(history, train_rows)[-1]
        np.testing.assert_allclose(
            actual, [0, 0, 0, 12, 12 * self.score, 0, 0, 0, 0], atol=1e-6
        )

    def test_last_window_excludes_old_posts_and_zero_scores(self):
        history, train_rows = self.history(
            [-1, 1, 0], ["alpha", "alpha", "unseenword"]
        )
        with patch.object(features, "N_LAST", 2):
            actual = features._text_bm25(history, train_rows)[-1]
        np.testing.assert_allclose(
            actual, [0, 0, 0, 0, 0, 0, 1, self.score, self.score], atol=1e-6
        )

    def test_same_timestamp_future_other_authors_and_current_resid_are_excluded(self):
        history, train_rows = self.history([-1, 0])
        current_row = len(history) - 1
        current = history.iloc[[-1]]
        extras = pd.concat(
            [
                current.assign(post="simultaneous", resid=2),
                current.assign(
                    post="future",
                    posted_at=current["posted_at"] + pd.Timedelta(days=1),
                    resid=2,
                ),
                current.assign(
                    author="other",
                    post="other",
                    posted_at=pd.Timestamp("2020-01-01", tz="UTC"),
                    resid=2,
                ),
            ],
            ignore_index=True,
        )
        expanded = pd.concat([history, extras], ignore_index=True)
        expected = [1, self.score, -self.score, 1, self.score, 0, 0, 0, 0]
        np.testing.assert_allclose(
            features._text_bm25(expanded, train_rows)[current_row], expected, atol=1e-6
        )
        expanded.loc[current_row, "resid"] = -999
        expanded.loc[current_row + 2, "text"] = "futureword"
        np.testing.assert_allclose(
            features._text_bm25(expanded, train_rows)[current_row], expected, atol=1e-6
        )

    def test_tied_matches_prefer_recent_posts_and_missing_residuals_are_skipped(self):
        history, train_rows = self.history([-1, np.nan, 1])
        with patch.object(features, "BM25_TOP_N", 1):
            actual = features._text_bm25(history, train_rows)[-1]
        np.testing.assert_allclose(
            actual, [0, 0, 0, 0, 0, 0, 1, self.score, self.score], atol=1e-6
        )
        # Set all target history residuals to missing without touching train.
        history.loc[history["author"] == "target", "resid"] = np.nan
        np.testing.assert_array_equal(
            features._text_bm25(history, train_rows)[-1], np.zeros(9)
        )

    def test_empty_query_missing_date_and_empty_training_vocabulary(self):
        history, train_rows = self.history([0], query="")
        np.testing.assert_array_equal(
            features._text_bm25(history, train_rows)[-1], np.zeros(9)
        )
        history.loc[history.index[-1], "text"] = "alpha"
        history.loc[history.index[-1], "posted_at"] = pd.NaT
        np.testing.assert_array_equal(
            features._text_bm25(history, train_rows)[-1], np.zeros(9)
        )
        history.loc[train_rows, "text"] = ""
        np.testing.assert_array_equal(
            features._text_bm25(history, train_rows), np.zeros((len(history), 9))
        )

    def test_prepare_bm25_only_preserves_rows_and_uses_numeric_preprocessing(self):
        history, _ = self.history([-1, 0, 1])
        history.index = np.arange(len(history)) * 11
        history = history.iloc[::-1]
        train = history.loc[history["author"] == "calibration", ["author", "post", "text"]]
        test = history.loc[history["post"] == "post_3", ["author", "post", "text"]]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(features, "_e5_embeddings", side_effect=AssertionError):
                with patch.object(features, "TfidfVectorizer", side_effect=AssertionError):
                    prepared_train, prepared_test = features.prepare(
                        train, test, features.TEXT_COMPARISON_BM25_NUM, Path(directory), history=history
                    )
        self.assertEqual(
            set(prepared_test.columns) - set(test.columns), set(features.TEXT_COMPARISON_BM25_NUM)
        )
        pd.testing.assert_index_equal(prepared_train.index, train.index)
        pd.testing.assert_index_equal(prepared_test.index, test.index)
        np.testing.assert_allclose(
            prepared_test[features.TEXT_COMPARISON_BM25_NUM],
            [[1, self.score, -self.score, 1, self.score, 0, 1, self.score, self.score]],
            atol=1e-6,
        )
        transformer = features.preprocess(features.TEXT_COMPARISON_BM25_NUM)
        self.assertEqual(transformer.fit_transform(prepared_train).shape, (len(train), 9))
        self.assertEqual(transformer.transform(prepared_test).shape, (len(test), 9))
        self.assertEqual(set(transformer.named_transformers_) - {"remainder"}, {"num"})


if __name__ == "__main__":
    unittest.main()
