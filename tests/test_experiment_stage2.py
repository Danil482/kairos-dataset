"""Stage-two target isolation and persisted prediction alignment contracts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from kairos import experiment_stage2 as stage2


def synthetic_bundle(directory):
    # Interleaved train/test rows make positional-index mistakes visible.
    records = []
    for author, split in [("A", "train"), ("X", "test"), ("B", "train"),
                          ("C", "train"), ("Y", "test"), ("D", "train"),
                          ("E", "train"), ("F", "train")]:
        for post in range(2):
            records.append({"author": author, "post": f"p{post}", "split": split,
                            "label": post if split == "train" else np.nan,
                            "content_text": "synthetic text",
                            "posted_at": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=post),
                            "log_reactions": 2.0, "level": 1.0})
    table = pd.DataFrame(records)
    history = table.drop(columns=["split", "label"]).copy()
    train_idx = np.flatnonzero(table.split.eq("train"))
    folds = [(train_idx[a], train_idx[b]) for a, b in GroupKFold(3).split(
        train_idx, groups=table.iloc[train_idx].author)]
    dense_path = Path(directory) / "embeddings.npy"
    np.save(dense_path, np.ones((len(history), 3), dtype=np.float32))
    return {"table": table, "history": history, "folds": folds,
            "dense_path": str(dense_path), "run_id": "synthetic-run",
            "config": {}, "groups": {}, "report": {}}


class Stage2IsolationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.bundle = synthetic_bundle(self.output)

    def _make_knn(self, fold, block="knn_all", reverse_keys=False):
        calls = []
        positions = {tuple(key): i for i, key in enumerate(
            self.bundle["table"][stage2.KEYS].itertuples(index=False, name=None))}

        def fake_knn(history, vectors, queries, bank, **settings):
            calls.append((queries.copy(), bank.copy(), settings.copy()))
            # Match the public KNN contract: keys in query order, fresh index.
            result = queries[stage2.KEYS].reset_index(drop=True).copy()
            result["knn_marker"] = [positions[tuple(key)] for key in
                                    result[stage2.KEYS].itertuples(index=False, name=None)]
            return result.iloc[::-1].reset_index(drop=True) if reverse_keys else result

        with patch("kairos.experiment_knn.knn_features", side_effect=fake_knn), \
             patch.object(stage2, "checked_dense", return_value=np.ones((len(positions), 3))):
            stage2.make_extra(self.bundle, self.output, fold, block, device="cpu")
        saved = joblib.load(self.output / stage2.feature_path(fold, block))
        return calls, saved

    def _assert_bank_contract(self, calls, fit, val, inner_folds):
        table = self.bundle["table"]
        fit_keys = set(table.iloc[fit][stage2.KEYS].itertuples(index=False, name=None))
        val_keys = set(table.iloc[val][stage2.KEYS].itertuples(index=False, name=None))
        seen_queries = []
        self.assertEqual(len(calls), inner_folds + 1)
        for queries, bank, _ in calls[:-1]:
            query_keys = set(queries[stage2.KEYS].itertuples(index=False, name=None))
            bank_keys = set(bank[stage2.KEYS].itertuples(index=False, name=None))
            self.assertTrue(set(queries.author).isdisjoint(bank.author))
            self.assertEqual(query_keys | bank_keys, fit_keys)
            self.assertTrue(query_keys.isdisjoint(bank_keys))
            self.assertTrue(bank_keys.isdisjoint(val_keys))
            seen_queries.extend(query_keys)
        self.assertEqual(len(seen_queries), len(fit_keys))
        self.assertEqual(set(seen_queries), fit_keys)
        queries, bank, _ = calls[-1]
        self.assertEqual(set(queries[stage2.KEYS].itertuples(index=False, name=None)), val_keys)
        self.assertEqual(set(bank[stage2.KEYS].itertuples(index=False, name=None)), fit_keys)
        for _, bank, _ in calls:
            self.assertTrue(bank.split.eq("train").all())
            self.assertTrue(bank.label.notna().all())

    def test_outer_validation_never_enters_inner_or_validation_banks(self):
        fit, val = stage2.indices(self.bundle, "1")
        calls, artifact = self._make_knn("1")
        self._assert_bank_contract(calls, fit, val, inner_folds=2)
        needed = np.concatenate([fit, val])
        np.testing.assert_array_equal(artifact["frame"].loc[needed, "knn_marker"], needed)
        self.assertTrue(artifact["frame"].loc[self.bundle["table"].split.eq("test")].isna().all().all())
        self.assertEqual(artifact["inner_folds"], 2)

    def test_final_uses_three_inner_folds_and_only_train_targets(self):
        fit, val = stage2.indices(self.bundle, "final")
        calls, artifact = self._make_knn("final", "knn180")
        self._assert_bank_contract(calls, fit, val, inner_folds=3)
        self.assertEqual(artifact["inner_folds"], 3)
        for _, _, settings in calls:
            self.assertEqual(settings["window_days"], 180)
            self.assertEqual(settings["k"], 20)
        needed = np.concatenate([fit, val])
        np.testing.assert_array_equal(artifact["frame"].loc[needed, "knn_marker"], needed)

    def test_equal_size_authors_keep_identical_inner_banks_after_row_permutation(self):
        original = self.bundle["table"]
        shuffled = original.sample(frac=1, random_state=42).reset_index(drop=True)
        all_authors = set(original.loc[original.split.eq("train"), "author"])

        def memberships(table, n_splits):
            fit = np.flatnonzero(table.split.eq("train"))
            result = []
            seen_queries = []
            for a, b in stage2.stable_inner_folds(table, fit, n_splits):
                bank = frozenset(table.iloc[fit[a]].author)
                query = frozenset(table.iloc[fit[b]].author)
                self.assertTrue(bank.isdisjoint(query))
                self.assertEqual(bank | query, all_authors)
                result.append((bank, query))
                seen_queries.extend(query)
            self.assertEqual(len(seen_queries), len(all_authors))
            self.assertEqual(set(seen_queries), all_authors)
            return result

        for n_splits in (2, 3):
            with self.subTest(inner_folds=n_splits):
                self.assertEqual(memberships(original, n_splits), memberships(shuffled, n_splits))

    def test_feature_key_misalignment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "alignment"):
            self._make_knn("0", reverse_keys=True)

    def test_prepare_builds_repeatable_author_disjoint_train_folds(self):
        bundle = self.bundle
        with patch.object(stage2, "load_experiment_data", return_value=(
                bundle["table"], bundle["history"], {}, {})), \
             patch.object(stage2, "load_frozen_embeddings", return_value=(
                 np.ones((len(bundle["history"]), 3)), {"model": "synthetic"})), \
             patch.object(stage2, "fingerprint", return_value={"source_sha256": {}, "versions": {}}):
            first = stage2.prepare(self.output, self.output / "first", {})
            second = stage2.prepare(self.output, self.output / "second", {})
        validation = []
        table = bundle["table"]
        train = set(np.flatnonzero(table.split.eq("train")))
        for index, ((fit, val), (other_fit, other_val)) in enumerate(zip(first["folds"], second["folds"])):
            np.testing.assert_array_equal(fit, other_fit)
            np.testing.assert_array_equal(val, other_val)
            self.assertTrue(set(table.iloc[fit].author).isdisjoint(table.iloc[val].author))
            self.assertEqual(set(fit) | set(val), train)
            self.assertTrue(set(fit).isdisjoint(val))
            got_fit, got_val = stage2.indices(first, str(index))
            np.testing.assert_array_equal(got_fit, fit)
            np.testing.assert_array_equal(got_val, val)
            validation.extend(val)
        self.assertEqual(len(validation), len(train))
        self.assertEqual(set(validation), train)

    def test_prepare_refuses_exposed_test_targets(self):
        table = self.bundle["table"].copy()
        table.loc[table.split.eq("test"), "label"] = 2
        with patch.object(stage2, "load_experiment_data", return_value=(
                table, self.bundle["history"], {}, {})), \
             patch.object(stage2, "load_frozen_embeddings", return_value=(np.ones((len(table), 3)), {})):
            with self.assertRaisesRegex(ValueError, "Test labels"):
                stage2.prepare(self.output, self.output / "exposed", {})

    def _prepare_explicit(self, fold_authors, subdirectory="explicit", table=None):
        fold_file = self.output / "configs" / "folds.json"
        fold_file.parent.mkdir(exist_ok=True)
        fold_file.write_text(json.dumps({"fold_authors": fold_authors}))
        table = self.bundle["table"] if table is None else table
        with patch.object(stage2, "ROOT", self.output), \
             patch.object(stage2, "load_experiment_data", return_value=(
                 table, self.bundle["history"], {}, {})), \
             patch.object(stage2, "load_frozen_embeddings", return_value=(
                 np.ones((len(self.bundle["history"]), 3)), {"model": "synthetic"})), \
             patch.object(stage2, "fingerprint", side_effect=lambda *_: {"source_sha256": {}, "versions": {}}), \
             patch.object(stage2, "GroupKFold", side_effect=AssertionError("Explicit folds must never be regenerated")):
            return stage2.prepare(self.output, self.output / subdirectory,
                                  {"folds_file": "configs/folds.json"})

    def test_prepare_uses_explicit_authors_without_regenerating_even_after_row_reorder(self):
        authors = {"0": ["F", "A"], "1": ["E", "C"], "2": ["B", "D"]}
        first = self._prepare_explicit(authors)
        second = self._prepare_explicit(authors, "reordered", self.bundle["table"].iloc[::-1])
        for bundle in (first, second):
            self.assertEqual(bundle["identity"]["fold_authors_sha256"], stage2.digest(authors))
            table = bundle["table"]
            for fold, (fit, val) in enumerate(bundle["folds"]):
                self.assertEqual(set(table.iloc[val].author), set(authors[str(fold)]))
                self.assertTrue(set(table.iloc[fit].author).isdisjoint(table.iloc[val].author))
                self.assertTrue(table.iloc[np.concatenate([fit, val])].split.eq("train").all())
                self.assertEqual(len(fit) + len(val), table.split.eq("train").sum())

    def test_explicit_folds_reject_overlap_missing_duplicate_and_nontrain_authors(self):
        cases = {
            "overlap": {"0": ["F", "A"], "1": ["E", "C", "A"], "2": ["B", "D"]},
            "missing": {"0": ["A"], "1": ["E", "C"], "2": ["B", "D"]},
            "duplicate": {"0": ["F", "A", "A"], "1": ["E", "C"], "2": ["B", "D"]},
            "test_author": {"0": ["F", "A", "X"], "1": ["E", "C"], "2": ["B", "D"]},
            "unknown_author": {"0": ["F", "A", "Z"], "1": ["E", "C"], "2": ["B", "D"]},
        }
        for name, authors in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                self._prepare_explicit(authors, name)

    def test_loading_prepared_bundle_rejects_changed_explicit_fold_assignments(self):
        authors = {"0": ["F", "A"], "1": ["E", "C"], "2": ["B", "D"]}
        self._prepare_explicit(authors)
        with patch.object(stage2, "ROOT", self.output), \
             patch.object(stage2, "fingerprint", return_value={"source_sha256": {}, "versions": {}}):
            original = stage2.load_bundle(self.output / "explicit")
            self.assertEqual(original["identity"]["fold_authors_sha256"], stage2.digest(authors))
            # Still a complete valid split, but no longer the prepared experiment.
            authors["0"][0], authors["1"][0] = authors["1"][0], authors["0"][0]
            (self.output / "configs/folds.json").write_text(json.dumps({"fold_authors": authors}))
            with self.assertRaisesRegex(ValueError, "folds changed"):
                stage2.load_bundle(self.output / "explicit")

    def _write_predictions(self):
        for fold, (_, val) in enumerate(self.bundle["folds"]):
            # Different row probabilities reveal a positional mix-up on collect.
            probabilities = np.column_stack([.1 + val / 100, .2 + 0 * val, .7 - val / 100])
            value = {"run_id": self.bundle["run_id"], "fold": str(fold), "val": val.copy(),
                     "probabilities": probabilities, "name": stage2.job_name("A", "none", "base")}
            stage2.atomic_dump(value, self.output / "jobs" / str(fold) / "A__none__base.joblib")

    def test_collect_restores_train_order_without_test_rows(self):
        self._write_predictions()
        probabilities, fold_ids = stage2.collect(self.output, self.bundle, "A", "none", "base")
        train_idx = np.flatnonzero(self.bundle["table"].split.eq("train"))
        np.testing.assert_allclose(probabilities[:, 0], .1 + train_idx / 100)
        for fold, (_, val) in enumerate(self.bundle["folds"]):
            np.testing.assert_array_equal(fold_ids[np.isin(train_idx, val)], fold)

    def test_collect_signals_absent_job_and_rejects_incomplete_rows(self):
        self.assertIsNone(stage2.collect(self.output, self.bundle, "A", "none", "base"))
        self._write_predictions()
        path = self.output / "jobs/0/A__none__base.joblib"
        value = joblib.load(path)
        value["val"] = value["val"][:-1]
        value["probabilities"] = value["probabilities"][:-1]
        stage2.atomic_dump(value, path)
        with self.assertRaises(ValueError):
            stage2.collect(self.output, self.bundle, "A", "none", "base")

    def test_collect_rejects_rows_assigned_to_the_wrong_outer_fold(self):
        self._write_predictions()
        # Coverage is still complete; only checking NaNs cannot detect this.
        first_path = self.output / "jobs/0/A__none__base.joblib"
        second_path = self.output / "jobs/1/A__none__base.joblib"
        first, second = joblib.load(first_path), joblib.load(second_path)
        first["val"], second["val"] = second["val"], first["val"]
        stage2.atomic_dump(first, first_path)
        stage2.atomic_dump(second, second_path)
        with self.assertRaises(ValueError):
            stage2.collect(self.output, self.bundle, "A", "none", "base")

    def test_collect_rejects_test_rows_in_oof(self):
        self._write_predictions()
        path = self.output / "jobs/0/A__none__base.joblib"
        value = joblib.load(path)
        value["val"][0] = np.flatnonzero(self.bundle["table"].split.eq("test"))[0]
        stage2.atomic_dump(value, path)
        with self.assertRaises(ValueError):
            stage2.collect(self.output, self.bundle, "A", "none", "base")


if __name__ == "__main__":
    unittest.main()
