"""Synthetic release checks; these fixtures are not real benchmark results."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import prepare_text
import text_baseline
from kairos.data import load_posts
from kairos.label import CLASS_NAMES, relabel


def make_release(path: Path) -> Path:
    """Write 30 authors x 25 posts: 24 train / 6 test, with every class per author.

    Returns the release directory. It can feed prepare_text, then validation and
    final-prediction commands. Its 600 labels are synthetic (480 train / 120 test).
    """
    (path / "authors").mkdir(parents=True)
    (path / "posts").mkdir()
    train_authors, test_authors = [], []
    for index in range(30):
        author = f"AUTHOR_{index:04d}"
        (train_authors if index < 24 else test_authors).append(author)
        base = 100 + index * 5
        reactions = [base] * 5 + [0, base, base * 100, base, base] * 4
        posts = []
        for number, count in enumerate(reactions):
            wording = ("quiet personal update" if count == 0 else
                       "major announcement celebration" if count > base else
                       "ordinary weekly progress update")
            text = f"{wording} ⟨PERSON_{index}⟩ ⟨ORG_{index}⟩ ⟨AUTHOR_{index:04d}⟩ #news"
            posts.append({
                "urn": f"POST_{index:04d}_{number:03d}",
                "author": {"code": author},
                "postedDateTimestamp": 1700000000000 + number * 86400000,
                "totalReactionCount": count,
                "text": "" if number == 5 else text,
                "contentType": "text",
            })
        (path / "posts" / f"{author}.json").write_text(json.dumps({
            "author": author, "crawledAt": "2025-01-01T00:00:00Z", "posts": posts,
        }), encoding="utf-8")
        (path / "authors" / f"{author}.json").write_text(json.dumps({
            "author": author, "followers": 1000 + index * 50,
        }), encoding="utf-8")

    # load_posts checks the release marker before labels can be generated.
    (path / "labels.csv").write_text("author,post,split,label,class\n", encoding="utf-8")
    labelled = relabel(load_posts(path), train_authors)
    labelled["split"] = labelled["author"].map(
        {**dict.fromkeys(train_authors, "train"), **dict.fromkeys(test_authors, "test")})
    labelled["class"] = labelled["label"].map(CLASS_NAMES)
    # Deliberately do not store labels in feed order: export must retain CSV order.
    labelled = labelled.iloc[::-1]
    labelled[["author", "post", "split", "label", "class"]].to_csv(
        path / "labels.csv", index=False)
    low, high = labelled.attrs["cutoffs"]
    (path / "split.json").write_text(json.dumps({
        "train": train_authors, "test": test_authors,
        "cutoffs_on_log_residual": {"low": low, "high": high},
    }), encoding="utf-8")
    if not labelled.groupby("author")["label"].nunique().eq(3).all():
        raise AssertionError("synthetic fixture must give every author all three classes")
    return path


class PrepareTextTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="sma-text-test-")
        self.addCleanup(self.temporary.cleanup)
        self.data = make_release(Path(self.temporary.name) / "release")
        self.output = Path(self.temporary.name) / "prepared"

    def labels(self) -> pd.DataFrame:
        return pd.read_csv(self.data / "labels.csv")

    def test_export_preserves_split_order_and_excludes_targets(self) -> None:
        manifest = prepare_text.prepare(self.data, self.output)
        source = self.labels()
        exported = {}
        for split, count in (("train", 480), ("test", 120)):
            path = self.output / f"{split}.csv.gz"
            table = exported[split] = pd.read_csv(path)
            expected = source[source["split"] == split].reset_index(drop=True)
            pd.testing.assert_frame_equal(table[["author", "post"]], expected[["author", "post"]])
            self.assertEqual(len(table), count)
            self.assertFalse({"reactions", "log_reactions", "resid", "class", "split"} & set(table))
            self.assertEqual(set(table) - {"label"}, set(prepare_text.EXPORT_COLUMNS))
            self.assertEqual(manifest["files"][split]["sha256"],
                             hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertNotIn("label", exported["test"])
        self.assertEqual(set(exported["train"]["label"]), {0, 1, 2})
        self.assertFalse(set(exported["train"]["author"]) & set(exported["test"]["author"]))
        self.assertTrue(exported["train"]["text"].str.contains("⟨PERSON⟩", regex=False).any())
        self.assertTrue(exported["train"]["text"].eq("__empty__").any())
        self.assertFalse(exported["train"]["text"].str.contains(r"(?:PERSON|ORG|AUTHOR)_\d+").any())
        self.assertEqual(manifest, json.loads((self.output / "manifest.json").read_text()))
        self.assertEqual(manifest, prepare_text.prepare(self.data, self.output))

    def test_rejects_labels_inconsistent_with_posts(self) -> None:
        labels = self.labels()
        labels.loc[0, "label"] = (labels.loc[0, "label"] + 1) % 3
        labels.to_csv(self.data / "labels.csv", index=False)
        with self.assertRaisesRegex(ValueError, "differs from the labels recomputed"):
            prepare_text.prepare(self.data, self.output)
        self.assertFalse(self.output.exists())

    def test_rejects_author_leaking_between_splits(self) -> None:
        labels = self.labels()
        labels.loc[0, "split"] = "train"
        labels.to_csv(self.data / "labels.csv", index=False)
        with self.assertRaisesRegex(ValueError, "authors present in both train and test"):
            prepare_text.prepare(self.data, self.output)

    def test_rejects_missing_feed(self) -> None:
        (self.data / "posts" / "AUTHOR_0000.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing from the feeds"):
            prepare_text.prepare(self.data, self.output)

    def test_rejects_missing_author_profile(self) -> None:
        (self.data / "authors" / "AUTHOR_0000.json").unlink()
        with self.assertRaisesRegex(ValueError, "profiles are missing"):
            prepare_text.prepare(self.data, self.output)

    def test_rejects_duplicate_post_keys(self) -> None:
        labels = self.labels()
        pd.concat([labels, labels.iloc[:1]]).to_csv(self.data / "labels.csv", index=False)
        with self.assertRaisesRegex(ValueError, r"duplicate \(author, post\) keys"):
            prepare_text.prepare(self.data, self.output)

    def test_rejects_invalid_labels_and_split(self) -> None:
        labels = self.labels()
        for column, value, message in (("label", 9, "labels must be 0, 1 or 2"),
                                       ("split", "validation", "both train and test")):
            with self.subTest(column=column):
                invalid = labels.copy()
                invalid.loc[0, column] = value
                invalid.to_csv(self.data / "labels.csv", index=False)
                with self.assertRaisesRegex(ValueError, message):
                    prepare_text.prepare(self.data, self.output)

    def test_missing_training_age_does_not_use_test_dates(self) -> None:
        path = self.data / "posts" / "AUTHOR_0000.json"
        feed = json.loads(path.read_text())
        feed["crawledAt"] = None
        path.write_text(json.dumps(feed))
        prepare_text.prepare(self.data, self.output)
        first = pd.read_csv(self.output / "train.csv.gz")
        self.assertTrue(first.loc[first.author == "AUTHOR_0000", "age_days"].isna().all())
        for index in range(24, 30):
            path = self.data / "posts" / f"AUTHOR_{index:04d}.json"
            feed = json.loads(path.read_text())
            feed["crawledAt"] = "2040-01-01T00:00:00Z"
            path.write_text(json.dumps(feed))
        prepare_text.prepare(self.data, self.output)
        pd.testing.assert_frame_equal(first, pd.read_csv(self.output / "train.csv.gz"))

    def test_validation_and_final_predictions_with_optional_embeddings(self) -> None:
        prepare_text.prepare(self.data, self.output)
        embedding_dir = Path(self.temporary.name) / "embeddings"
        embedding_dir.mkdir()
        rng = np.random.default_rng(42)
        metadata = {"input_sha256": {}, "output_sha256": {}, "rows": {}, "dimensions": 384}
        for split, count in (("train", 480), ("test", 120)):
            values = rng.normal(size=(count, 384)).astype(np.float32)
            values /= np.linalg.norm(values, axis=1, keepdims=True)
            path = embedding_dir / f"{split}.npy"
            np.save(path, values)
            metadata["rows"][split] = count
            metadata["input_sha256"][split] = text_baseline.sha256(self.output / f"{split}.csv.gz")
            metadata["output_sha256"][split] = text_baseline.sha256(path)
        (embedding_dir / "metadata.json").write_text(json.dumps(metadata))
        for optional_embeddings in (None, embedding_dir):
            with self.subTest(embeddings=optional_embeddings is not None):
                train, test, columns, fingerprints = text_baseline.read_inputs(
                    self.output, optional_embeddings)
                config = text_baseline.configuration(fingerprints, columns)
                results = Path(self.temporary.name) / ("neural" if columns else "cpu")
                results.mkdir()
                text_baseline.validate(train, columns, config, results)
                report = json.loads((results / "validation.json").read_text())
                self.assertTrue(set(report["validation_authors"]) <= set(train.author))
                self.assertFalse(set(report["validation_authors"]) & set(test.author))
                held_out = train[train.author.isin(report["validation_authors"])]
                self.assertEqual(report["n_validation"], len(held_out))
                text_baseline.final(train, test, columns, config, results)
                predictions = pd.read_csv(results / "predictions.csv")
                pd.testing.assert_frame_equal(predictions[["author", "post"]], test[["author", "post"]])
                self.assertTrue(predictions.label.isin([0, 1, 2]).all())
                with self.assertRaisesRegex(ValueError, "different data, code or versions"):
                    text_baseline.final(train, test, columns, {**config, "seed": 99}, results)
        metadata["input_sha256"]["test"] = "wrong file order"
        (embedding_dir / "metadata.json").write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "different prepared files"):
            text_baseline.read_inputs(self.output, embedding_dir)


if __name__ == "__main__":
    unittest.main()
