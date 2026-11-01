"""Encoder boundary checks without torch, model downloads, or training."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from kairos import experiment_encode as encoder


class FakeTokenizer:
    """One token per whitespace-delimited word; exact IDs preserve source order."""
    def __init__(self):
        self.vocabulary = {}

    def encode(self, text, add_special_tokens=False):
        result = []
        for word in text.split():
            result.append(self.vocabulary.setdefault(word, len(self.vocabulary) + 10))
        return result

    def __call__(self, texts, **kwargs):
        return {"input_ids": [self.encode(text) for text in texts]}

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def prepare_for_model(self, ids, **kwargs):
        values = [1] + list(ids) + [2]
        return {"input_ids": values, "attention_mask": [1] * len(values)}


def frame():
    return pd.DataFrame([
        {"author": "AUTHOR_1", "post": "POST_1", "text": "hello world",
         "content_text": "hello world title", "shared_text": "", "article_title": "title",
         "document_title": "", "document_text": ""},
        {"author": "AUTHOR_1", "post": "POST_2", "text": "second post",
         "content_text": "second post", "shared_text": "", "article_title": "",
         "document_title": "", "document_text": ""},
    ])


class TokenBudgetTest(unittest.TestCase):
    def test_each_source_has_reserved_budget_and_total_includes_prefix(self):
        tokenizer = FakeTokenizer()
        row = ("own " * 300, "combined must not be encoded", "shared " * 300,
               "title " * 100, "", "document " * 300)
        features, stats = encoder._prepare_input_batch([row], tokenizer, 512)
        self.assertEqual(len(features[0]["input_ids"]), 512)
        self.assertEqual(stats["kept_tokens"],
                         {"own": 256, "shared": 160, "titles": 64, "document": 29})
        self.assertEqual(stats["truncated_rows"], 1)
        self.assertEqual(stats["truncated_by_source"], dict.fromkeys(encoder.SOURCE_NAMES, 1))
        self.assertNotIn("combined", tokenizer.vocabulary)
        self.assertEqual(features[0]["input_ids"][1], tokenizer.vocabulary["query:"])

    def test_unused_budget_flows_forward_and_empty_inputs_get_a_token(self):
        tokenizer = FakeTokenizer()
        rows = [("own " * 10, "", "shared " * 400, "title " * 10, "", "doc " * 100),
                ("", "", "", "", "", "")]
        features, stats = encoder._prepare_input_batch(rows, tokenizer, 512)
        self.assertEqual(len(features[0]["input_ids"]), 512)
        self.assertEqual(stats["kept_tokens"],
                         {"own": 11, "shared": 400, "titles": 10, "document": 89})
        self.assertEqual(stats["truncated_rows"], 1)
        self.assertIn(tokenizer.vocabulary["__empty__"], features[1]["input_ids"])

    def test_smaller_limit_cannot_overflow_or_drop_prefix(self):
        tokenizer = FakeTokenizer()
        features, stats = encoder._prepare_input_batch(
            [("own " * 100, "", "shared " * 100, "", "", "doc " * 100)], tokenizer, 32)
        self.assertEqual(len(features[0]["input_ids"]), 32)
        self.assertEqual(stats["kept_tokens"]["own"], 29)
        with self.assertRaisesRegex(ValueError, "no room"):
            encoder._prepare_input_batch([("",) * 6], tokenizer, 3)


class CacheTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="sma-encoder-test-")
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.history = frame()
        self.expected = {"input": encoder.history_fingerprint(self.history),
                         "config": {"device": "cpu"}, "model": encoder.MODEL,
                         "revision": encoder.REVISION, "runtime": {"compute_dtype": "float32"}}
        self.vectors = encoder._normalise(np.ones((2, 384)), 2)

    def test_cache_binding_catches_reordered_rows_source_and_config_changes(self):
        encoder._save_cache(self.output, self.vectors, self.expected)
        loaded = encoder._load_cache(self.output, self.expected)
        self.assertIsNotNone(loaded)
        np.testing.assert_array_equal(loaded[0], self.vectors)
        reordered = encoder.history_fingerprint(self.history.iloc[::-1])
        self.assertNotEqual(reordered["ordered_keys_sha256"], self.expected["input"]["ordered_keys_sha256"])
        self.assertIsNone(encoder._load_cache(self.output, {**self.expected, "input": reordered}))
        changed = self.history.copy()
        changed.loc[0, "shared_text"] = "different source"
        self.assertIsNone(encoder._load_cache(
            self.output, {**self.expected, "input": encoder.history_fingerprint(changed)}))
        self.assertIsNone(encoder._load_cache(
            self.output, {**self.expected, "config": {"device": "cuda"}}))
        self.assertIsNone(encoder._load_cache(
            self.output, {**self.expected, "runtime": {"compute_dtype": "bfloat16"}}))

    def test_partial_corrupt_and_nonunit_cache_are_rejected(self):
        self.assertIsNone(encoder._load_cache(self.output, self.expected))
        encoder._save_cache(self.output, self.vectors, self.expected)
        (self.output / "metadata.json").unlink()
        self.assertIsNone(encoder._load_cache(self.output, self.expected))
        encoder._save_cache(self.output, self.vectors, self.expected)
        with (self.output / "history.npy").open("ab") as stream:
            stream.write(b"corruption")
        self.assertIsNone(encoder._load_cache(self.output, self.expected))
        encoder._save_cache(self.output, self.vectors * 2, self.expected)
        self.assertIsNone(encoder._load_cache(self.output, self.expected))
        (self.output / "metadata.json").write_text("broken json")
        self.assertIsNone(encoder._load_cache(self.output, self.expected))
        (self.output / "metadata.json").write_text("[]")
        self.assertIsNone(encoder._load_cache(self.output, self.expected))

    def test_labels_do_not_enter_fingerprint_but_keys_must_be_unique(self):
        self.history["label"] = [0, 2]
        self.history["reactions"] = [3, 10000]
        self.assertEqual(encoder.history_fingerprint(self.history), self.expected["input"])
        duplicate = pd.concat([self.history, self.history.iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            encoder.history_fingerprint(duplicate)

    def test_float32_renormalisation_rejects_invalid_vectors(self):
        values = encoder._normalise(np.full((2, 384), 0.049, dtype=np.float16), 2)
        self.assertEqual(values.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-5)
        with self.assertRaisesRegex(ValueError, "zero"):
            encoder._normalise(np.zeros((2, 384)), 2)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            encoder._normalise(np.full((2, 384), np.nan), 2)


if __name__ == "__main__":
    unittest.main()
