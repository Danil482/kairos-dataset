"""Protect long-document coverage and post-to-vector alignment."""
import unittest
import numpy as np
import pandas as pd

from scripts.semantic_embeddings import chunk_token_ids, collect_texts


class SemanticEmbeddingTests(unittest.TestCase):
    def test_short_document_has_all_tokens_in_order(self):
        ids = list(range(1000))
        chunks = chunk_token_ids(ids)
        self.assertEqual([token for chunk in chunks for token in chunk], ids)
        self.assertLessEqual(max(map(len, chunks)), 480)

    def test_long_document_retains_beginning_middle_and_end(self):
        chunks = chunk_token_ids(list(range(3000)))
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[1][0], 1440)
        self.assertEqual(chunks[-1][-1], 2999)

    def test_deduplication_preserves_empty_posts_and_both_text_views(self):
        table = pd.DataFrame(dict(text=["hello", " ", "hello"], expanded_text=["hello", "context", "hello context"]))
        unique, mappings, clipped = collect_texts(table)
        self.assertEqual(unique, ["hello", "context", "hello context"])
        np.testing.assert_array_equal(mappings["text"], [0, -1, 0])
        np.testing.assert_array_equal(mappings["expanded_text"], [0, 1, 2])
        self.assertEqual(clipped, 0)


if __name__ == "__main__":
    unittest.main()
