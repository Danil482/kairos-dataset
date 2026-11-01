"""Exact permitted-bank KNN contracts, without training a model."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from kairos.experiment_knn import KNN_COLUMNS, knn_features


def fixture():
    history = pd.DataFrame({
        "author": ["A", "B", "B", "C", "D", "A", "E", "F"],
        "post": ["old", "b2", "b1", "c1", "future", "query", "same_time", "empty"],
        "posted_at": pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-02",
                                     "2020-01-04", "2020-01-11", "2020-01-10",
                                     "2020-01-10", "2020-01-05"], utc=True),
        "content_text": ["text"] * 7 + [""],
        "log_reactions": [999, 5, 3, 8, 999, 999, 999, 999],
        "level": [0, 1, 1, 2, 0, 0, 0, 0],
    })
    vectors = np.array([[1, 0], [1, 0], [1, 0], [.6, .8], [1, 0], [1, 0],
                        [1, 0], [1, 0]], dtype=np.float32)
    bank = history[["author", "post"]].assign(label=[2, 1, 0, 2, 2, 2, 2, 2])
    return history, vectors, history.iloc[[5]], bank


class ExactKnnTests(unittest.TestCase):
    def test_manual_weighted_result_filters_then_caps_authors(self):
        history, vectors, queries, bank = fixture()
        row = knn_features(history, vectors, queries, bank, k=2).iloc[0]
        # A is excluded; D is future; E is simultaneous; F has empty text.
        # B has tied vectors: b1 wins lexically, although it is older than b2.
        self.assertEqual(row.knn_neighbor_count, 2)
        np.testing.assert_allclose([row.knn_fraction_0, row.knn_fraction_1, row.knn_fraction_2],
                                   [1 / 1.6, 0, .6 / 1.6], atol=1e-7)
        self.assertAlmostEqual(row.knn_mean_residual, (2 + .6 * 6) / 1.6, places=6)
        self.assertAlmostEqual(row.knn_mean_cosine, .8, places=6)
        self.assertEqual(row.knn_max_cosine, 1)
        self.assertEqual(row.knn_mean_age_days, 7)
        self.assertEqual(row.knn_missing, 0)

    def test_author_cap_is_applied_before_topk_and_window_is_inclusive(self):
        history, vectors, queries, bank = fixture()
        # b1 is 8d old and excluded, b2 is 7d old and included at the boundary.
        row = knn_features(history, vectors, queries, bank, k=2, window_days=7).iloc[0]
        self.assertEqual(row.knn_neighbor_count, 2)
        self.assertAlmostEqual(row.knn_fraction_1, 1 / 1.6, places=6)
        self.assertEqual(row.knn_mean_age_days, 6.5)

    def test_tie_between_authors_uses_stable_keys_not_bank_row_order(self):
        history, vectors, queries, bank = fixture()
        vectors[3] = [1, 0]
        first = knn_features(history, vectors, queries, bank, k=1)
        shuffled = knn_features(history, vectors, queries, bank.iloc[::-1], k=1)
        assert_frame_equal(first, shuffled)
        self.assertEqual(first.iloc[0].knn_fraction_0, 1)

    def test_timestamp_storage_resolution_does_not_change_age_or_window(self):
        history, vectors, queries, bank = fixture()
        expected = knn_features(history, vectors, queries, bank, k=2, window_days=7)
        history["posted_at"] = history.posted_at.dt.as_unit("us")
        actual = knn_features(history, vectors, queries, bank, k=2, window_days=7)
        assert_frame_equal(actual, expected)

    def test_query_order_and_batch_boundaries_do_not_change_features(self):
        history, vectors, _, bank = fixture()
        queries = history.iloc[[5, 3, 1, 7]]
        first = knn_features(history, vectors, queries, bank, k=2, batch_size=1)
        second = knn_features(history, vectors, queries, bank, k=2, batch_size=128)
        assert_frame_equal(first, second, check_exact=False, atol=1e-7, rtol=1e-7)
        assert_frame_equal(first[["author", "post"]], queries[["author", "post"]].reset_index(drop=True))

    def test_empty_bank_or_no_eligible_neighbors_has_explicit_missing_values(self):
        history, vectors, queries, bank = fixture()
        for subset in (bank.iloc[:0], bank.iloc[[0, 4, 5, 6, 7]]):
            with self.subTest(rows=len(subset)):
                row = knn_features(history, vectors, queries, subset).iloc[0]
                self.assertTrue(row[KNN_COLUMNS[:7]].isna().all())
                self.assertEqual(row.knn_neighbor_count, 0)
                self.assertEqual(row.knn_missing, 1)

    def test_labels_outside_selected_permitted_neighbors_are_irrelevant(self):
        history, vectors, queries, bank = fixture()
        before = knn_features(history, vectors, queries, bank, k=2)
        # These labels cannot participate: own author, future, equal-time, empty.
        bank.loc[[0, 4, 5, 6, 7], "label"] = 0
        queries = queries.assign(label=0)
        history["label"] = 2  # No implicit history target lookup is allowed.
        history.loc[[0, 4, 5, 6, 7], "log_reactions"] = -100
        after = knn_features(history, vectors, queries, bank, k=2)
        assert_frame_equal(before, after)
        # Changing the selected b1 bank label must change its class contribution.
        bank.loc[2, "label"] = 1
        changed = knn_features(history, vectors, queries, bank, k=2)
        self.assertEqual(changed.iloc[0].knn_fraction_0, 0)
        self.assertGreater(changed.iloc[0].knn_fraction_1, 0)

    def test_explicit_bank_cannot_silently_expand(self):
        history, vectors, queries, bank = fixture()
        row = knn_features(history, vectors, queries, bank.iloc[[3]], k=20).iloc[0]
        self.assertEqual(row.knn_neighbor_count, 1)
        self.assertEqual(row.knn_fraction_2, 1)
        self.assertEqual(row.knn_mean_residual, 6)

    def test_zero_and_invalid_vectors_are_not_neighbors(self):
        history, vectors, queries, bank = fixture()
        vectors[1:3] = 0
        vectors[3] = np.nan
        row = knn_features(history, vectors, queries, bank).iloc[0]
        self.assertEqual(row.knn_neighbor_count, 0)
        vectors[5] = 0
        self.assertEqual(knn_features(history, vectors, queries, bank).iloc[0].knn_missing, 1)

    def test_nonpositive_cosines_do_not_invent_target_fallback(self):
        history, vectors, queries, bank = fixture()
        vectors[1:3] = [-1, 0]
        vectors[3] = [0, 1]
        row = knn_features(history, vectors, queries, bank, k=2).iloc[0]
        self.assertEqual(row.knn_neighbor_count, 2)
        self.assertEqual(row.knn_missing, 1)
        self.assertTrue(row[KNN_COLUMNS[:4]].isna().all())
        self.assertAlmostEqual(row.knn_mean_cosine, -.5)

    def test_bad_bank_keys_and_labels_fail(self):
        history, vectors, queries, bank = fixture()
        for invalid in (bank.assign(label=3), pd.concat([bank, bank.iloc[[0]]]),
                        bank.assign(post="absent")):
            with self.assertRaises(ValueError):
                knn_features(history, vectors, queries, invalid)

    def test_cuda_matches_cpu_when_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        history, vectors, _, bank = fixture()
        queries = history.iloc[[5, 3, 1]]
        before = torch.backends.cuda.matmul.allow_tf32
        gpu = knn_features(history, vectors, queries, bank, k=2, device="cuda", batch_size=2)
        cpu = knn_features(history, vectors, queries, bank, k=2, device="cpu", batch_size=2)
        assert_frame_equal(cpu, gpu, check_exact=False, rtol=1e-6, atol=1e-6)
        self.assertEqual(before, torch.backends.cuda.matmul.allow_tf32)


if __name__ == "__main__":
    unittest.main()
