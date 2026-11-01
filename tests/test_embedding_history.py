from __future__ import annotations

import unittest
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from lab1_embeddings import EmbeddingStore, causal_context


class EmbeddingHistoryTests(unittest.TestCase):
    def test_future_text_embeddings_do_not_change_past_context(self):
        vectors = np.random.default_rng(42).normal(size=(12, 6)).astype(np.float32)
        original = causal_context(vectors)
        vectors[8:] *= -10
        changed = causal_context(vectors)
        np.testing.assert_allclose(original[:8], changed[:8], equal_nan=True)
        self.assertTrue(np.isnan(original[0, :3]).all())
        np.testing.assert_array_equal(original[:, 3], np.arange(12))

    def test_initial_history_and_mixed_precision_timestamps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "encoder.json").write_text(json.dumps({"history_input_sha256": "fixture"}))
            pd.DataFrame({"author": ["A", "A"], "post": ["P7", "P6"]}).to_csv(root / "posts.csv", index=False)
            pd.DataFrame({"author": ["A"]}).to_csv(root / "profiles.csv", index=False)
            np.save(root / "posts.npy", np.array([[0, 1], [1, 0]], dtype=np.float32))
            np.save(root / "profiles.npy", np.array([[1, 0]], dtype=np.float32))
            np.save(root / "history.npy", np.tile([1., 0.], (5, 1)))
            pd.DataFrame({"author": ["A"] * 5, "post": [f"P{i}" for i in range(1, 6)],
                          "posted_at": ["2025-01-01 00:00:00.123+00:00", "2025-01-02 00:00:00+00:00",
                                        "2025-01-03 00:00:00+00:00", "2025-01-04 00:00:00+00:00",
                                        "2025-01-05 00:00:00+00:00"],
                          "history_order": range(5)}).to_csv(root / "history.csv", index=False)
            table = pd.DataFrame({"author": ["A", "A"], "post": ["P7", "P6"], "hist_n": [6, 5],
                                  "posted_at": pd.to_datetime(["2025-01-07", "2025-01-06"], utc=True)})
            result = EmbeddingStore(root).context(table)
            self.assertEqual(result.loc[1, "e5_previous_similarity"], 1)
            self.assertEqual(result.loc[1, "e5_history_posts"], 5)
            self.assertEqual(result.loc[0, "e5_previous_similarity"], 0)
            self.assertEqual(result.loc[0, "e5_history_posts"], 6)


if __name__ == "__main__":
    unittest.main()
