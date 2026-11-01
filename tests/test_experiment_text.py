"""Synthetic causal and fit/transform contracts; no real-data training."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.base import clone

from kairos.experiment_text import (
    fit_svd, history_feature_columns, history_vector_features, make_text_model,
)


def history_fixture():
    frame = pd.DataFrame({
        "author": ["A", "A", "A", "A", "B"],
        "post": ["p0", "p1", "p2", "p3", "other"],
        "posted_at": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03",
                                     "2020-01-04", "2020-01-01"], utc=True),
        "content_text": ["first", "second", "third", "fourth", "other author"],
        "log_reactions": [1.0, 5.0, 99.0, 6.0, 1000.0],
        "level": [np.nan, 1.0, 3.0, 5.0, np.nan],
    })
    vectors = np.array([[1, 0], [0, 1], [1, 0], [1, 1], [1, 0]], dtype=np.float32)
    return frame, vectors


class HistoryVectorTests(unittest.TestCase):
    def test_current_and_other_author_are_not_neighbors(self):
        frame, vectors = history_fixture()
        features = history_vector_features(frame, vectors, k=1)
        self.assertEqual(features.loc[0, "semantic_valid_history_count"], 0)
        self.assertEqual(features.loc[1, "semantic_cosine_max"], 0)
        self.assertEqual(features.loc[2, "semantic_cosine_last"], 0)
        self.assertEqual(features.loc[2, "semantic_cosine_max"], 1)
        self.assertEqual(features.loc[2, "semantic_valid_history_count"], 2)
        self.assertAlmostEqual(features.loc[2, "semantic_neighbor_log_reactions_mean_minus_level"], -2)
        self.assertEqual(features.loc[4, "semantic_valid_history_count"], 0)

    def test_current_reactions_and_labels_do_not_change_own_features(self):
        frame, vectors = history_fixture()
        before = history_vector_features(frame, vectors)
        frame.loc[2, "log_reactions"] = 10000
        frame["label"] = [2, 0, 1, 2, 1]
        after = history_vector_features(frame, vectors)
        assert_frame_equal(before.iloc[:3], after.iloc[:3])
        self.assertNotEqual(before.loc[3, "semantic_neighbor_log_reactions_mean_minus_level"],
                            after.loc[3, "semantic_neighbor_log_reactions_mean_minus_level"])

    def test_future_append_does_not_change_existing_features(self):
        frame, vectors = history_fixture()
        before = history_vector_features(frame, vectors)
        future = frame.iloc[[2]].assign(post="future", posted_at=pd.Timestamp("2030-01-01", tz="UTC"),
                                       log_reactions=100000)
        after = history_vector_features(pd.concat([frame, future], ignore_index=True),
                                        np.vstack([vectors, [1, 0]]))
        assert_frame_equal(before, after.iloc[:len(frame)].reset_index(drop=True))

    def test_equal_timestamps_excluded_and_input_order_preserved(self):
        frame, vectors = history_fixture()
        frame.loc[1, "posted_at"] = frame.loc[2, "posted_at"]
        expected = history_vector_features(frame, vectors, k=1)
        self.assertEqual(expected.loc[2, "semantic_valid_history_count"], 1)
        order = [3, 0, 4, 2, 1]
        shuffled = history_vector_features(frame.iloc[order], vectors[order], k=1)
        assert_frame_equal(expected.iloc[order].reset_index(drop=True), shuffled)

    def test_empty_oov_and_nonfinite_do_not_create_neighbors(self):
        frame, vectors = history_fixture()
        frame.loc[0, "content_text"] = "__empty__"
        vectors[1] = 0  # A nonempty text entirely outside the fitted vocabulary.
        vectors[2] = np.nan
        features = history_vector_features(frame, vectors)
        self.assertTrue(features["semantic_valid_history_count"].eq(0).all())
        self.assertTrue(features["semantic_missing"].eq(1).all())
        self.assertTrue(features["semantic_cosine_max"].isna().all())
        self.assertTrue(features["semantic_neighbor_log_reactions_mean_minus_level"].isna().all())

    def test_counts_survive_a_missing_current_text_and_nonunit_vectors(self):
        frame, vectors = history_fixture()
        frame.loc[2, "content_text"] = "  "
        result = history_vector_features(frame, vectors * 7)
        self.assertEqual(result.loc[2, "semantic_valid_history_count"], 2)
        self.assertEqual(result.loc[2, "semantic_missing"], 1)
        self.assertTrue(np.isnan(result.loc[2, "semantic_cosine_max"]))
        self.assertAlmostEqual(result.loc[3, "semantic_cosine_past_mean"], 1, places=6)

    def test_column_groups_separate_historical_outcomes(self):
        similarity, outcome = history_feature_columns("lexical")
        self.assertEqual(len(similarity), 7)
        self.assertEqual(outcome, ["lexical_neighbor_log_reactions_mean_minus_level"])
        frame, vectors = history_fixture()
        actual = history_vector_features(frame, vectors, prefix="lexical")
        self.assertEqual(list(actual), ["author", "post"] + similarity + outcome)

    def test_recent_centroid_and_topk_ties_have_defined_meanings(self):
        frame = pd.DataFrame({
            "author": ["A"] * 7, "post": [f"p{i}" for i in range(7)],
            "posted_at": pd.date_range("2020-01-01", periods=7, tz="UTC"),
            "text": ["nonempty"] * 7, "log_reactions": [0, 1, 2, 3, 4, 5, 99],
            "level": [3.0] * 7,
        })
        vectors = np.array([[0, 1]] + [[1, 0]] * 6, dtype=np.float32)
        last = history_vector_features(frame, vectors, k=3).iloc[-1]
        self.assertAlmostEqual(last.semantic_cosine_past_mean, 5 / np.sqrt(26))
        self.assertEqual(last.semantic_cosine_last5_mean, 1)
        self.assertEqual(last.semantic_cosine_topk_mean, 1)
        # Five equally close candidates: the latest three have log counts 3,4,5.
        self.assertEqual(last.semantic_neighbor_log_reactions_mean_minus_level, 1)


class TextTransformerTests(unittest.TestCase):
    def test_svd_fits_only_supplied_text_and_transforms_unseen_words_to_zero(self):
        texts = pd.Series(["apple pear fruit", "pear fruit", "apple orange", "orange pear"])
        model = fit_svd(texts, {"min_df": 1, "ngram_range": [1, 1], "n_components": 128})
        transformed = model.transform(texts)
        self.assertEqual(transformed.shape, (4, 4))
        self.assertEqual(transformed.dtype, np.float32)
        before = dict(model.named_steps["tfidf"].vocabulary_)
        np.testing.assert_array_equal(model.transform(["unseenonlyword"]), np.zeros((1, 4)))
        self.assertEqual(before, model.named_steps["tfidf"].vocabulary_)
        self.assertIsNotNone(clone(model))

    def test_one_feature_fixture_and_unnormalized_projection(self):
        one = fit_svd(["apple", "apple"], {"min_df": 1, "ngram_range": [1, 1]})
        self.assertEqual(one.transform(["apple", "unknown"]).shape, (2, 1))
        low_rank = fit_svd(["apple pear", "pear orange", "orange banana", "banana apple"],
                          {"min_df": 1, "ngram_range": [1, 1], "n_components": 1})
        norms = np.linalg.norm(low_rank.transform(["apple pear", "banana"]), axis=1)
        self.assertFalse(np.allclose(norms, 1))

    def test_runner_flat_svd_settings(self):
        model = fit_svd(["apple pear", "pear orange", "orange banana", "banana apple"],
                        {"min_df": 1, "svd_components": 2, "svd_n_iter": 3, "seed": 7})
        self.assertEqual(model.named_steps["svd"].n_components_, 2)
        self.assertEqual(model.named_steps["svd"].n_iter, 3)
        self.assertEqual(model.named_steps["svd"].random_state, 7)

    def test_word_and_character_text_pipelines_on_tiny_fixture(self):
        texts = pd.Series(["quiet update", "quiet note", "normal day", "normal work",
                           "great achievement", "great celebration"])
        labels = [0, 0, 1, 1, 2, 2]
        for analyzer in ("word", "char"):
            with self.subTest(analyzer=analyzer):
                model = make_text_model({"tfidf": {"analyzer": analyzer, "min_df": 1},
                                         "classifier": {"max_iter": 100}})
                model.fit(texts, labels)
                self.assertEqual(model.predict_proba(texts).shape, (6, 3))
                self.assertEqual(model.named_steps["clf"].class_weight, "balanced")


if __name__ == "__main__":
    unittest.main()
