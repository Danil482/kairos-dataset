import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from kairos import features
from kairos.data import load_followers, load_posts
from kairos.label import author_level
from kairos_lab.acquire import archive_entries
from kairos_lab.common import Stopped, mark, stop_check, verified_marker
from kairos_lab.dataset import FORBIDDEN, audit_release_labels, history_features, prepare
from kairos_lab.encoders import VectorCache, contextual_embeddings, media_mapping
from kairos_lab.train import score_final, train_models
from kairos_lab.verify import verify
from tests.fixtures import make_release


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_strict_history_excludes_current_future_other_author_and_ties(self):
        data = self.root / "release"
        make_release(data)
        posts = load_posts(data)
        # Give two posts the same timestamp; neither may enter the other's past.
        posts.loc[posts.index[8], "posted_at"] = posts.loc[posts.index[7], "posted_at"]

        def build(p):
            return history_features(features.build(author_level(p), load_followers(data)))

        original = build(posts)
        changed = posts.copy()
        changed.loc[changed.index[7:24], "reactions"] = 99999999
        changed.loc[changed.index[24:], "reactions"] = 88888888
        perturbed = build(changed)
        columns = [c for c in original.select_dtypes(include="number") if c not in FORBIDDEN]
        # The modified current post, tied post, and all earlier posts keep their inputs.
        pd.testing.assert_frame_equal(original.loc[:8, columns], perturbed.loc[:8, columns])
        self.assertEqual(original.loc[7, "history_count"], 7)
        self.assertEqual(original.loc[8, "history_count"], 7)

    def test_contextual_vectors_use_only_prior_outcomes(self):
        posts = pd.DataFrame({"author": ["A", "A", "A", "B"],
                              "posted_at": pd.to_datetime([1, 2, 2, 1], unit="s"), "level": [0., 1., 1., 0.]})
        vectors = np.array([[1., 0.], [.5, .5], [0., 1.], [1., 0.]], dtype=np.float32)
        first = contextual_embeddings(posts, vectors, np.array([10, 20, 30, 40]))
        second = contextual_embeddings(posts, vectors, np.array([10, 9999, 8888, 7777]))
        np.testing.assert_array_equal(first, second)

    def test_artifact_identity_and_hash_are_required_for_resume(self):
        artifact = self.root / "result"
        artifact.write_text("good")
        marker = self.root / "done.json"
        mark(marker, "abc", [artifact])
        self.assertTrue(verified_marker(marker, "abc"))
        with self.assertRaises(RuntimeError):
            verified_marker(marker, "changed")
        artifact.write_text("corrupt")
        with self.assertRaises(RuntimeError):
            verified_marker(marker, "abc")

    def test_label_precision_fallback_still_requires_exact_classes_and_keys(self):
        data = self.root / "release"
        make_release(data)
        posts = load_posts(data)
        from kairos.data import load_labels, load_split
        from kairos.label import relabel
        labels, split = load_labels(data), load_split(data)
        original = relabel(posts, split["train"])
        perturbed = original.copy()
        perturbed.iloc[0, perturbed.columns.get_loc("label")] = (int(perturbed.iloc[0].label) + 1) % 3
        with patch("kairos_lab.dataset.relabel", return_value=perturbed):
            if np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant:
                result = audit_release_labels(posts, labels, split["train"])
                self.assertTrue(result["exact"])
                self.assertEqual(result["log1p_precision"], "longdouble_rounded_to_float64")
            else:
                with self.assertRaisesRegex(ValueError, "extended precision unavailable"):
                    audit_release_labels(posts, labels, split["train"])
        corrupted = labels.copy()
        corrupted.loc[0, "label"] = (int(corrupted.loc[0, "label"]) + 1) % 3
        with self.assertRaises(ValueError):
            audit_release_labels(posts, corrupted, split["train"])
        with self.assertRaises(ValueError):
            audit_release_labels(posts, labels.iloc[:-1], split["train"])

    def test_stop_is_never_removed(self):
        stop = self.root / "STOP"
        stop.touch()
        with self.assertRaises(Stopped):
            stop_check(self.root)
        self.assertTrue(stop.exists())

    def test_archive_traversal_and_links_are_rejected(self):
        for entry in ("Path = ../escape\nSize = 1", "Path = /root/escape", "Path = data/link\nSymbolic Link = /etc/passwd"):
            with self.assertRaises(ValueError):
                archive_entries("metadata\n----------\n" + entry + "\n\n")
        self.assertEqual(archive_entries("x\n----------\nPath = data/posts/file.json\nSize = 2\n\n")[0]["Size"], "2")

    def test_vector_cache_identity_and_resume(self):
        path = self.root / "vectors.sqlite"
        cache = VectorCache(path, "same")
        cache.put("text", np.array([1., 2.]))
        cache.commit(); cache.close()
        cache = VectorCache(path, "same")
        np.testing.assert_array_equal(cache.get("text"), [1., 2.])
        cache.close()
        with self.assertRaises(ValueError):
            VectorCache(path, "different")

    def test_media_join_rejects_avatars_and_reshared_files(self):
        from PIL import Image
        media = self.root / "media"
        media.mkdir()
        for name in ["post.png", "avatar.png", "reshared.png"]:
            Image.new("RGB", (2, 2)).save(media / name)
        rows = [{"author": "A", "post": "P", "files": [
                   {"role": "image", "path": "media/post.png"},
                   {"role": "avatar", "path": "media/avatar.png"}]},
                {"author": "A", "post": "Q", "path": "media/reshared.png", "from_reshared_post": True}]
        (media / "manifest.jsonl").write_text("\n".join(map(json.dumps, rows)))
        mapping, _ = media_mapping(self.root, 4)
        self.assertEqual(mapping[("A", "P")], [media / "post.png"])
        self.assertNotIn(("A", "Q"), mapping)

    def test_end_to_end_synthetic_release_resume_and_final_boundary(self):
        data, work = self.root / "release", self.root / "run"
        config = make_release(data)
        prepare(data, work, config)
        train = pd.read_parquet(work / "cache/train.parquet")
        test = pd.read_parquet(work / "cache/test.parquet")
        self.assertNotIn("label", test)
        self.assertNotIn("reactions", train)
        self.assertEqual(train.groupby("author").fold.nunique().max(), 1)
        train_models(work, config)
        predictions = (work / "predictions/final.csv").read_bytes()
        train_models(work, config)
        self.assertEqual(predictions, (work / "predictions/final.csv").read_bytes())
        # A test-label edit cannot influence training or model selection.
        original_labels = (data / "labels.csv").read_bytes()
        labels = pd.read_csv(data / "labels.csv")
        labels.loc[labels.split == "test", "label"] = 2
        labels.to_csv(data / "labels.csv", index=False)
        train_models(work, config)
        self.assertEqual(predictions, (work / "predictions/final.csv").read_bytes())
        (data / "labels.csv").write_bytes(original_labels)
        score_final(data, work, config)
        verify(work, config)
        result = json.loads((work / "report/verification.json").read_text())
        self.assertEqual(result["cache_replay"], "passed")
        output = work / "predictions/final.csv"
        changed = pd.read_csv(output)
        changed.loc[0, "label"] = (int(changed.loc[0, "label"]) + 1) % 3
        changed.to_csv(output, index=False)
        with self.assertRaises(RuntimeError):
            score_final(data, work, config)


if __name__ == "__main__":
    unittest.main()
