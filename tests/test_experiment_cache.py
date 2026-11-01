"""Prediction-cache recovery and atomic persistence, without training a model."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from kairos import experiment as exp
from kairos import experiment_cache as frozen
from kairos.experiment_encode import history_fingerprint


class ExperimentCacheTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sma-cache-test-")
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.table = pd.DataFrame([
            {"author": f"A{author}", "post": f"P{author}_{label}",
             "split": "train", "label": label}
            for author in range(10) for label in range(3)
        ])
        self.history = self.table[["author", "post"]].copy()
        self.config = {"seed": 42, "validation_fraction": .2, "text_enabled": False}
        self.cache = self.output / "validation_probabilities" / "E0_baseline.npz"

    def run_validation(self, manifest=None):
        # This oracle makes cached-vs-recomputed results observable; no estimator runs.
        def predict(recipe, context, fit_indices, prediction_indices):
            labels = context.table.iloc[prediction_indices].label.to_numpy(dtype=int)
            return np.eye(3)[labels], None

        recipe = {"E0_baseline": {"kind": "tabular", "columns": []}}
        with patch.object(exp, "tabular_recipes", return_value=recipe), \
                patch.object(exp, "fit_predict", side_effect=predict) as fitter, \
                contextlib.redirect_stdout(io.StringIO()):
            report = exp.validate(self.table, self.history, {}, self.config,
                                  self.output, manifest or {"fixture": 1})
        return report, fitter.call_count

    def prime_cache(self):
        report, calls = self.run_validation()
        self.assertEqual(calls, 1)
        with np.load(self.cache, allow_pickle=False) as archive:
            probabilities = archive["probabilities"].copy()
            tag = archive["tag"].item()
        return report, probabilities, tag

    def test_valid_probability_cache_avoids_refitting(self):
        original, _, _ = self.prime_cache()
        repeated, calls = self.run_validation()
        self.assertEqual(calls, 0)
        self.assertEqual(original["selected"], repeated["selected"])
        self.assertEqual(original["results"]["E0_baseline"]["confusion_matrix"],
                         repeated["results"]["E0_baseline"]["confusion_matrix"])

    def test_invalid_probability_values_are_recomputed(self):
        _, correct, tag = self.prime_cache()
        invalid = {}
        for name, value in (("nan", np.nan), ("infinity", np.inf),
                            ("negative", -.01), ("over_one", 1.01)):
            values = correct.copy()
            values[0, 0] = value
            invalid[name] = values
        invalid["wrong_sum"] = correct * .5
        invalid["wrong_shape"] = correct[:-1]
        invalid["strings"] = correct.astype(str)
        invalid["complex"] = correct.astype(complex)
        for name, values in invalid.items():
            with self.subTest(name=name):
                np.savez_compressed(self.cache, probabilities=values, tag=tag)
                _, calls = self.run_validation()
                self.assertEqual(calls, 1)
                with np.load(self.cache, allow_pickle=False) as repaired:
                    np.testing.assert_array_equal(repaired["probabilities"], correct)

    def test_wrong_tag_and_missing_fields_are_cache_misses(self):
        _, correct, tag = self.prime_cache()
        for fields in ({"probabilities": correct, "tag": "stale"},
                       {"tag": tag}, {"probabilities": correct}):
            with self.subTest(fields=list(fields)):
                np.savez_compressed(self.cache, **fields)
                _, calls = self.run_validation()
                self.assertEqual(calls, 1)

    def test_malformed_truncated_and_corrupt_compressed_files_are_recomputed(self):
        self.prime_cache()
        original = self.cache.read_bytes()
        corrupted = bytearray(original)
        with zipfile.ZipFile(io.BytesIO(original)) as archive:
            info = archive.getinfo("probabilities.npy")
            offset = info.header_offset + 30 + len(info.filename.encode())
            # The local header may contain a Zip64 extra field; read its length.
            extra_length = int.from_bytes(original[info.header_offset + 28:info.header_offset + 30], "little")
            offset += extra_length
        corrupted[offset:offset + 4] = b"\xff\xff\xff\xff"
        for content in (b"", b"not a NumPy archive", original[:len(original) // 2], bytes(corrupted)):
            with self.subTest(length=len(content)):
                self.cache.write_bytes(content)
                _, calls = self.run_validation()
                self.assertEqual(calls, 1)
                with np.load(self.cache, allow_pickle=False) as repaired:
                    self.assertTrue(np.isfinite(repaired["probabilities"]).all())

    def test_interrupted_save_preserves_previous_cache_and_removes_temporary(self):
        self.prime_cache()
        original = self.cache.read_bytes()

        def interrupted_write(stream, **arrays):
            stream.write(b"partial new archive")
            raise OSError("synthetic write failure")

        with patch.object(exp.np, "savez_compressed", side_effect=interrupted_write):
            with self.assertRaisesRegex(OSError, "synthetic write failure"):
                self.run_validation({"fixture": 2})
        self.assertEqual(self.cache.read_bytes(), original)
        self.assertEqual(list(self.cache.parent.iterdir()), [self.cache])


class FrozenEmbeddingsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sma-frozen-cache-test-")
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.history = pd.DataFrame([
            {"author": "A1", "post": f"P{i}", "text": f"own {i}",
             "content_text": f"combined {i}", "shared_text": "shared",
             "article_title": "article", "document_title": "document", "document_text": "body"}
            for i in range(3)
        ])
        vectors = np.random.default_rng(42).normal(size=(3, 384)).astype(np.float32)
        self.vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        self.metadata = {
            **frozen.FROZEN_IDENTITY,
            "input": history_fingerprint(self.history),
            "config": {"device": "cuda", "max_length": 512, "bf16": True, "batch_size": 64},
            "runtime": {"compute_dtype": "bfloat16", "versions": {"torch": "producer-only-version"}},
        }
        self.write_cache()

    def write_cache(self, vectors=None):
        np.save(self.output / "history.npy", self.vectors if vectors is None else vectors,
                allow_pickle=False)
        self.metadata["output_sha256"] = hashlib.sha256((self.output / "history.npy").read_bytes()).hexdigest()
        self.write_metadata()

    def write_metadata(self):
        (self.output / "metadata.json").write_text(json.dumps(self.metadata))

    def test_cpu_consumption_preserves_provenance_and_is_read_only(self):
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.output.iterdir()}
        with patch("importlib.metadata.version", side_effect=AssertionError("must not inspect consumer versions")):
            actual, metadata = frozen.load_frozen_embeddings(self.history, self.output)
        np.testing.assert_array_equal(actual, self.vectors)
        self.assertEqual(metadata, self.metadata)
        self.assertFalse(actual.flags.writeable)
        with self.assertRaises(ValueError):
            actual[0, 0] = 0
        after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.output.iterdir()}
        self.assertEqual(before, after)

    def test_import_and_load_do_not_import_torch_or_sentence_transformers(self):
        script = """import builtins, json, sys
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'sentence_transformers'}:
        raise AssertionError('neural dependency imported: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
import pandas as pd
from pathlib import Path
from kairos.experiment_cache import load_frozen_embeddings
vectors, metadata = load_frozen_embeddings(pd.DataFrame(json.loads(sys.argv[1])), Path(sys.argv[2]))
assert vectors.shape == (3, 384)
assert 'torch' not in sys.modules and 'sentence_transformers' not in sys.modules
"""
        subprocess.run([sys.executable, "-c", script, self.history.to_json(orient="records"),
                        str(self.output)], check=True, capture_output=True, text=True,
                       cwd=Path(__file__).resolve().parents[1])

    def test_changed_order_or_text_cannot_use_existing_vectors(self):
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            frozen.load_frozen_embeddings(self.history.iloc[::-1], self.output)
        for column in ("text", "shared_text", "content_text", "document_text"):
            changed = self.history.copy()
            changed.loc[0, column] += " changed"
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                frozen.load_frozen_embeddings(changed, self.output)

    def test_changed_identity_and_length_are_rejected(self):
        changes = {"revision": "different", "model": "another/model", "dimensions": 768,
                   "prefix": "passage: ", "token_budgets": {"own": 512},
                   "encoder_sha256": "0" * 64}
        for key, value in changes.items():
            original = self.metadata[key]
            self.metadata[key] = value
            self.write_metadata()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, f"producer mismatch: {key}"):
                frozen.load_frozen_embeddings(self.history, self.output)
            self.metadata[key] = original
        self.metadata["config"]["max_length"] = 256
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "max_length=512"):
            frozen.load_frozen_embeddings(self.history, self.output)

    def test_missing_or_corrupt_cache_raises_without_repair(self):
        for filename in ("history.npy", "metadata.json"):
            self.write_cache()
            (self.output / filename).unlink()
            with self.subTest(filename=filename), self.assertRaisesRegex(FileNotFoundError, "incomplete"):
                frozen.load_frozen_embeddings(self.history, self.output)
            self.assertFalse((self.output / filename).exists())
        self.write_cache()
        with (self.output / "history.npy").open("ab") as stream:
            stream.write(b"changed bytes")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            frozen.load_frozen_embeddings(self.history, self.output)
        (self.output / "metadata.json").write_text("[]")
        with self.assertRaisesRegex(ValueError, "JSON object"):
            frozen.load_frozen_embeddings(self.history, self.output)

    def test_checksum_alone_does_not_accept_invalid_arrays(self):
        for name, values, message in (
                ("dtype", self.vectors.astype(np.float64), "dtype"),
                ("shape", self.vectors[:, :-1], "shape"),
                ("nan", np.full_like(self.vectors, np.nan), "non-finite"),
                ("zero", np.zeros_like(self.vectors), "unit-normalised"),
                ("norm", self.vectors * 2, "unit-normalised")):
            self.write_cache(values)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                frozen.load_frozen_embeddings(self.history, self.output)


if __name__ == "__main__":
    unittest.main()
