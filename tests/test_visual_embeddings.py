"""Protect content-only filtering, deduplication and missing-image alignment."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.visual_embeddings import select_content_rows, aggregate_posts, media_path


class VisualEmbeddingTests(unittest.TestCase):
    def test_only_allowed_training_content_and_unique_post_files(self):
        row = dict(author="a", post="p", file="x.jpg", kind="post_image", content_type="image/jpeg", from_reshared_post=False)
        rows = [row, row, {**row, "from_reshared_post":True}, {**row, "kind":"article_image"},
                {**row, "kind":"author_picture", "file":"avatar.jpg"},
                {**row, "kind":"company_logo", "file":"logo.jpg"}, {**row, "author":"held_out"}]
        pairs, excluded = select_content_rows(rows, {("a", "p")})
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["kinds"], ["article_image", "post_image"])
        self.assertTrue(pairs[0]["own"] and pairs[0]["shared"])
        self.assertEqual(excluded, {"author_picture":1, "company_logo":1})

    def test_missing_and_failed_images_do_not_drop_or_shift_posts(self):
        table = pd.DataFrame(dict(author=["a", "b", "c"], post=["p", "q", "r"]))
        pair = dict(author="a", post="p", kinds=["post_image"], own=True, shared=False)
        index = {"pairs":[{**pair, "image_index":0}, {**pair, "image_index":1},
                           {**pair, "author":"c", "post":"r", "image_index":2}]}
        vectors = np.array([[1.,0.], [0.,1.], [0.,0.]], dtype=np.float32)
        details = np.array([[100,200,1000,1], [300,200,2000,1], [0,0,0,0]], dtype=np.float32)
        mean, first, meta = aggregate_posts(table, index, vectors, details)
        np.testing.assert_allclose(mean[0], [2**-.5,2**-.5], atol=1e-6)
        np.testing.assert_array_equal(first[0], [1,0])
        np.testing.assert_array_equal(mean[1:], np.zeros((2,2)))
        np.testing.assert_array_equal(meta.visual_available, [1,0,0])
        np.testing.assert_array_equal(meta.visual_count, [2,0,1])
        np.testing.assert_array_equal(meta.visual_decoded_count, [2,0,0])

    def test_media_paths_cannot_escape_dataset(self):
        root = Path.cwd() / "data" / "media"
        with self.assertRaises(ValueError):
            media_path("../../outside.jpg", root)
        self.assertEqual(media_path("a/image.jpg", root), (root / "a" / "image.jpg").resolve())


if __name__ == "__main__":
    unittest.main()
