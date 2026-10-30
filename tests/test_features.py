"""Feature preparation contracts: train-only fitting, row alignment, and selection."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from kairos import features


class FeaturePreparationTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame(
            {
                "text": ["alpha beta", "alpha gamma", "beta gamma", "alpha beta gamma"],
                "n_chars": [10, 11, 10, 17],
                "content_type": ["text"] * 4,
            },
            index=[9, 2, 6, 1],
        )
        self.test = pd.DataFrame(
            {
                "text": ["onlyintest", "alpha"],
                "n_chars": [10, 5],
                "content_type": ["unknown", "text"],
            },
            index=[12, 4],
        )
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.results = Path(directory.name)

    def test_tfidf_fits_train_only_and_preserves_rows(self):
        with patch.object(features, "_e5_embeddings", side_effect=AssertionError):
            train, test = features.prepare(
                self.train, self.test, features.TEXT_TFIDF_NUM, self.results
            )
            changed_test = self.test.assign(text=["alpha alpha alpha", "beta beta"])
            repeated_train, _ = features.prepare(
                self.train, changed_test, features.TEXT_TFIDF_NUM, self.results
            )

        pd.testing.assert_index_equal(train.index, self.train.index)
        pd.testing.assert_index_equal(test.index, self.test.index)
        pd.testing.assert_frame_equal(train, repeated_train)
        columns = features.TEXT_TFIDF_NUM[:5]
        self.assertEqual(
            set(train.columns) - set(self.train.columns), set(columns)
        )
        self.assertTrue(
            all(
                isinstance(dtype, pd.SparseDtype)
                for dtype in train[columns].dtypes
            )
        )
        # A test-only word must never enter the vocabulary or get a nonzero weight.
        self.assertEqual(test.loc[12, columns].sparse.to_dense().sum(), 0)
        transformer = features.preprocess(features.TEXT_TFIDF_NUM)
        self.assertEqual(transformer.fit_transform(train).shape, (len(train), 5))
        self.assertEqual(transformer.transform(test).shape, (len(test), 5))

    def test_svd_only_selects_exactly_the_text_vector(self):
        with patch.object(features, "_e5_embeddings", side_effect=AssertionError):
            train, test = features.prepare(
                self.train, self.test, features.TEXT_SVD_NUM, self.results
            )
        transformer = features.preprocess(features.TEXT_SVD_NUM)
        matrix = transformer.fit_transform(train)
        predicted_matrix = transformer.transform(test)
        columns = features.TEXT_SVD_NUM[:4]
        self.assertEqual(set(train.columns) - set(self.train.columns), set(columns))
        self.assertEqual(matrix.shape, (len(train), 4))
        np.testing.assert_array_equal(matrix, train[columns].to_numpy())
        np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-6)
        self.assertEqual(predicted_matrix.shape, (len(test), 4))
        np.testing.assert_array_equal(predicted_matrix[0], np.zeros(4))
        self.assertNotIn("cat", transformer.named_transformers_)
        self.assertNotIn("num", transformer.named_transformers_)

    def test_missing_fixed_columns_are_errors(self):
        for columns in (["missing_numeric"], features.TEXT_E5_NUM):
            with self.subTest(columns=columns):
                with self.assertRaises(ValueError):
                    features.preprocess(columns).fit(self.train)

    def test_preprocessing_supports_both_classifiers_and_sparse_tfidf(self):
        columns = ["n_chars", "content_type"] + features.TEXT_TFIDF_NUM
        train, test = features.prepare(self.train, self.test, columns, self.results)
        transformer = features.preprocess(columns)
        matrix = transformer.fit_transform(train)
        test_matrix = transformer.transform(test)
        self.assertTrue(sparse.issparse(matrix))
        self.assertTrue(sparse.issparse(test_matrix))
        # Unknown test categories must not change the train-fitted encoding.
        self.assertEqual(matrix.shape[1], test_matrix.shape[1])
        labels = [0, 1, 2, 1]
        logistic = LogisticRegression(max_iter=1000).fit(matrix, labels)
        boosting = HistGradientBoostingClassifier(max_iter=2, random_state=42).fit(
            matrix.toarray(), labels
        )
        self.assertEqual(logistic.predict(test_matrix).shape, (len(test),))
        self.assertEqual(boosting.predict(test_matrix.toarray()).shape, (len(test),))

    def test_e5_keeps_text_and_row_order(self):
        embeddings = np.arange(6 * 384, dtype=np.float32).reshape(6, 384)
        with patch.object(features, "_e5_embeddings", return_value=embeddings) as encode:
            train, test = features.prepare(
                self.train, self.test, features.TEXT_E5_NUM, self.results
            )
        encode.assert_called_once_with(
            self.train["text"].tolist() + self.test["text"].tolist(), self.results
        )
        np.testing.assert_array_equal(train[features.TEXT_E5_NUM], embeddings[:4])
        np.testing.assert_array_equal(test[features.TEXT_E5_NUM], embeddings[4:])
        pd.testing.assert_index_equal(train.index, self.train.index)
        pd.testing.assert_index_equal(test.index, self.test.index)

    def test_ordinary_features_do_not_prepare_text(self):
        with patch.object(features, "TfidfVectorizer", side_effect=AssertionError):
            with patch.object(features, "_e5_embeddings", side_effect=AssertionError):
                train, test = features.prepare(
                    self.train.drop(columns="text"),
                    self.test.drop(columns="text"),
                    ["n_chars"],
                    self.results,
                )
        self.assertNotIn("text", train)
        self.assertNotIn("text", test)


if __name__ == "__main__":
    unittest.main()
