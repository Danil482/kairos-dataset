"""History comparisons: cosine formulas, time boundaries, and feature preparation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

from kairos import features


class TextComparisonTests(unittest.TestCase):
    @staticmethod
    def history(residuals):
        return pd.DataFrame(
            {
                "author": ["a"] * len(residuals),
                "posted_at": pd.date_range("2025-01-01", periods=len(residuals), tz="UTC"),
                "resid": residuals,
            }
        )

    def test_cosine_of_mean_and_weighted_sum_for_dense_and_sparse_vectors(self):
        history = self.history([2, -3, 999])
        vectors = np.array([[1, 0], [0, 2], [2, 0]], dtype=np.float32)
        expected = [[0, 0, 0], [0, 0, 0], [1 / np.sqrt(5), 1 / np.sqrt(5), 2]]
        # The mean of [1, 0] and [0, 2] is [0.5, 1], not a mean of pairwise cosines.
        with patch.object(features, "N_LAST", 2), patch.object(features, "N_NEAREST", 2):
            for matrix in (vectors, sparse.csr_matrix(vectors)):
                with self.subTest(sparse=sparse.issparse(matrix)):
                    actual = features._text_comparison(history, matrix)
                    np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_all_history_and_last_five_are_different_windows(self):
        history = self.history([7, 0, 0, 0, 0, 0, 999])
        vectors = np.array([[1, 0]] + [[0, 1]] * 5 + [[1, 0]], dtype=np.float32)
        with patch.object(features, "N_LAST", 6), patch.object(features, "N_NEAREST", 2):
            actual = features._text_comparison(history, vectors)
        np.testing.assert_allclose(actual[-1], [1 / np.sqrt(26), 0, 7], atol=1e-6)

    def test_same_timestamp_future_and_other_authors_are_excluded(self):
        history = self.history([999, 1000, 2, 100, 2000])
        history["author"] = ["a", "b", "a", "a", "a"]
        history["posted_at"] = pd.to_datetime(
            ["2025-01-02", "2025-01-01", "2025-01-01", "2025-01-02", "2025-01-03"],
            utc=True,
        )
        vectors = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
        actual = features._text_comparison(history, vectors)
        np.testing.assert_allclose(actual[0], [1, 1, 2])
        np.testing.assert_array_equal(actual[1], np.zeros(3))
        np.testing.assert_array_equal(actual[3], np.zeros(3))

        changed = history.copy()
        changed.loc[[0, 3, 4], "resid"] = -9999
        changed_vectors = vectors.copy()
        changed_vectors[4] = [1, 0]
        repeated = features._text_comparison(changed, changed_vectors)
        np.testing.assert_array_equal(repeated[:4], actual[:4])

    def test_missing_residuals_are_filtered_within_the_last_window(self):
        history = self.history([10, np.nan, 3, 999])
        vectors = np.ones((4, 2), dtype=np.float32)
        with patch.object(features, "N_LAST", 2), patch.object(features, "N_NEAREST", 2):
            actual = features._text_comparison(history, vectors)
        # The old residual 10 is outside the last two posts and cannot replace NaN.
        np.testing.assert_allclose(actual[-1], [1, 1, 3], atol=1e-6)

        history["resid"] = np.nan
        actual = features._text_comparison(history, vectors)
        np.testing.assert_array_equal(actual[:, 2], np.zeros(4))

    def test_negative_cosines_zero_vectors_and_missing_dates(self):
        history = self.history([2, 999, 999, 999])
        history.loc[3, "posted_at"] = pd.NaT
        vectors = np.array([[1, 0], [-1, 0], [0, 0], [1, 0]], dtype=np.float32)
        with patch.object(features, "N_LAST", 1), patch.object(features, "N_NEAREST", 1):
            actual = features._text_comparison(history, vectors)
        np.testing.assert_allclose(actual[1], [-1, -1, -2])
        np.testing.assert_array_equal(actual[2:], np.zeros((2, 3)))

    def test_equal_cosines_prefer_the_more_recent_post(self):
        history = self.history([2, 5, 999])
        vectors = np.ones((3, 2), dtype=np.float32)
        with patch.object(features, "N_LAST", 2), patch.object(features, "N_NEAREST", 1):
            actual = features._text_comparison(history, vectors)
        np.testing.assert_allclose(actual[-1, 2], 5, atol=1e-6)

    def test_window_sizes_must_be_positive_integers_and_ordered(self):
        for n_last, n_nearest in ((0, 1), (2, 3), (2.5, 1), (True, 1)):
            with self.subTest(n_last=n_last, n_nearest=n_nearest):
                with patch.object(features, "N_LAST", n_last):
                    with patch.object(features, "N_NEAREST", n_nearest):
                        with self.assertRaises(ValueError):
                            features._text_comparison(
                                self.history([0]), np.ones((1, 2), dtype=np.float32)
                            )


class ComparisonPreparationTests(unittest.TestCase):
    def setUp(self):
        self.history = pd.DataFrame(
            {
                "author": ["a", "a", "a", "b", "b", "b", "unrelated"],
                "post": ["old", "first", "second", "old", "first", "second", "old"],
                "posted_at": pd.to_datetime(
                    ["2025-01-01", "2025-01-02", "2025-01-03"] * 2 + ["2025-01-01"],
                    utc=True,
                ),
                "text": ["alpha beta"] * 3 + ["onlyhistory"] * 3 + ["unused"],
                "resid": [np.nan, 2, 3, np.nan, 4, 999, 1000],
            },
            index=[91, 20, 84, 36, 53, 17, 10],
        ).iloc[[5, 0, 6, 2, 4, 1, 3]]
        self.train = self.history.loc[[84, 20], ["author", "post", "text"]].copy()
        self.test = self.history.loc[[17], ["author", "post", "text"]].copy()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.results = Path(directory.name)

    def test_tfidf_uses_train_vocabulary_and_unlabelled_author_history(self):
        columns = features.TEXT_TFIDF_NUM + features.TEXT_COMPARISON_TFIDF_NUM
        with patch.object(features, "_e5_embeddings", side_effect=AssertionError):
            with patch.object(features, "TruncatedSVD", side_effect=AssertionError):
                train, test = features.prepare(
                    self.train, self.test, columns, self.results, history=self.history
                )
        pd.testing.assert_index_equal(train.index, self.train.index)
        pd.testing.assert_index_equal(test.index, self.test.index)
        self.assertNotIn("resid", train)
        self.assertEqual(
            set(train.columns) - set(self.train.columns),
            set(features.TEXT_TFIDF_NUM[:3] + features.TEXT_COMPARISON_TFIDF_NUM),
        )
        np.testing.assert_allclose(
            train[features.TEXT_COMPARISON_TFIDF_NUM], [[1, 1, 2], [1, 1, 0]], atol=1e-6
        )
        # Neither test words nor words confined to unlabelled history enter the vocabulary.
        np.testing.assert_array_equal(
            test[features.TEXT_TFIDF_NUM[:3]].sparse.to_dense(), np.zeros((1, 3))
        )
        np.testing.assert_array_equal(
            test[features.TEXT_COMPARISON_TFIDF_NUM], np.zeros((1, 3))
        )

    def test_e5_is_encoded_once_and_shared_with_comparisons(self):
        history = self.history.copy()
        history["text"] = history["author"] + "/" + history["post"]
        train_input = self.train.assign(text=history.loc[self.train.index, "text"])
        test_input = self.test.assign(text=history.loc[self.test.index, "text"])
        source = history[history["author"] != "unrelated"]
        vectors = np.arange(len(source) * 384, dtype=np.float32).reshape(-1, 384) + 1
        columns = features.TEXT_E5_NUM + features.TEXT_COMPARISON_E5_NUM
        with patch.object(features, "_e5_embeddings", return_value=vectors) as encode:
            with patch.object(features, "TfidfVectorizer", side_effect=AssertionError):
                train, test = features.prepare(
                    train_input, test_input, columns, self.results, history=history
                )
        encode.assert_called_once_with(source["text"].tolist(), self.results)
        train_rows = source.index.get_indexer(self.train.index)
        test_rows = source.index.get_indexer(self.test.index)
        np.testing.assert_array_equal(train[features.TEXT_E5_NUM], vectors[train_rows])
        np.testing.assert_array_equal(test[features.TEXT_E5_NUM], vectors[test_rows])
        transformer = features.preprocess(features.TEXT_COMPARISON_E5_NUM)
        self.assertEqual(transformer.fit_transform(train).shape, (len(train), 3))
        self.assertEqual(transformer.transform(test).shape, (len(test), 3))
        self.assertEqual(set(transformer.named_transformers_) - {"remainder"}, {"num"})

    def test_comparison_requires_complete_unique_history(self):
        columns = features.TEXT_COMPARISON_TFIDF_NUM
        with self.assertRaisesRegex(ValueError, "history="):
            features.prepare(self.train, self.test, columns, self.results)
        with self.assertRaisesRegex(ValueError, "missing from history"):
            features.prepare(
                self.train,
                self.test,
                columns,
                self.results,
                history=self.history.drop(index=84),
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            features.prepare(
                self.train,
                self.test,
                columns,
                self.results,
                history=pd.concat([self.history, self.history.iloc[:1]]),
            )


if __name__ == "__main__":
    unittest.main()
