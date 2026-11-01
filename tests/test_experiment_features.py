"""Source extraction and causal-history contracts, without model training."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from kairos import features
from kairos.experiment_features import (CATEGORICAL, DOCUMENT_MAX_CHARS, FEATURE_GROUPS,
                                        NUMERIC, build_features, extract_post,
                                        load_experiment_data)
from kairos.label import author_level
from test_text_pipeline import make_release


def history_fixture(author="AUTHOR_1", reactions=None, text="ordinary text"):
    reactions = [9] * 5 + [99] * 5 + [9] if reactions is None else reactions
    rows = []
    for i, count in enumerate(reactions):
        rows.append(extract_post(author, {
            "urn": f"{author}_POST_{i}", "postedDateTimestamp": 1700000000000 + i * 86400000,
            "totalReactionCount": count, "text": text, "contentType": "text",
        }, "2025-01-01T00:00:00Z"))
    result = pd.DataFrame(rows)
    result["crawled_at"] = pd.to_datetime(result.crawled_at, utc=True)
    return result


class ExperimentFeatureContractTest(unittest.TestCase):
    def test_current_reaction_changes_no_feature_of_current_post(self):
        history = history_fixture()
        before = build_features(history)
        changed = history.copy()
        changed.loc[10, "reactions"] = 10_000_000
        changed["level"] = -1000.0  # A supplied level must not be trusted either.
        after = build_features(changed)
        pd.testing.assert_series_equal(before.iloc[10], after.iloc[10])
        self.assertFalse({"reactions", "log_reactions", "resid", "label", "class"} & set(before))

    def test_future_post_changes_no_earlier_feature(self):
        history = history_fixture()
        history.loc[3, "crawled_at"] = pd.NaT
        before = build_features(history)
        future = history.iloc[[-1]].copy()
        future["post"] = "FUTURE_POST"
        future["posted_at"] += pd.Timedelta(days=500)
        future["crawled_at"] += pd.Timedelta(days=3000)
        future["reactions"] = 100_000_000
        future["n_chars"] = 99999
        future["person_codes"] = [("PERSON_NEW",)]
        after = build_features(pd.concat([future, history], ignore_index=True))
        pd.testing.assert_frame_equal(before, after.iloc[:-1].reset_index(drop=True))
        self.assertTrue(pd.isna(after.loc[3, "age_days"]))

    def test_person_and_org_novelty_stays_inside_owner_and_precedes_update(self):
        a = history_fixture("AUTHOR_A", [9, 9], "⟨PERSON_1⟩ ⟨ORG_7⟩")
        b = history_fixture("AUTHOR_B", [9, 9], "⟨PERSON_1⟩ ⟨ORG_7⟩")
        built = build_features(pd.concat([b, a], ignore_index=True))
        for author in ["AUTHOR_A", "AUTHOR_B"]:
            rows = built[built.author == author]
            self.assertEqual(rows.person_new_count.tolist(), [1, 0])
            self.assertEqual(rows.person_seen_fraction.tolist(), [0, 1])
            self.assertEqual(rows.org_new_count.tolist(), [1, 0])
            self.assertEqual(rows.org_seen_fraction.tolist(), [0, 1])
        self.assertNotIn("PERSON_1", " ".join(built.content_text))

    def test_recent_windows_use_the_known_earlier_values(self):
        built = build_features(history_fixture())
        self.assertAlmostEqual(built.loc[10, "recent5_delta"], math.log(10) / 2)
        self.assertAlmostEqual(built.loc[10, "recent10_delta"], 0)
        self.assertAlmostEqual(built.loc[10, "recent5_vs_previous5"], math.log(10))
        self.assertEqual(built.loc[5, "recent5_delta"], 0)
        self.assertTrue(pd.isna(built.loc[9, "recent10_delta"]))
        self.assertEqual(built.loc[9, "has_10_history"], 0)
        self.assertEqual(built.loc[10, "has_10_history"], 1)
        self.assertAlmostEqual(built.loc[10, "gap_surprise"], 0)

    def test_original_baseline_numbers_preserved_before_text_normalisation(self):
        raw_text = "  Hi ⟨PERSON_12345⟩! ⟨URL⟩\n#topic?  "
        history = history_fixture(text=raw_text)
        followers = pd.Series({"AUTHOR_1": 1500})
        original = features.build(author_level(history.assign(text=raw_text)), followers)
        actual = build_features(history, followers)
        for name in FEATURE_GROUPS["baseline"]:
            np.testing.assert_allclose(actual[name], original[name], equal_nan=True,
                                       err_msg=f"baseline feature differs: {name}")
        self.assertNotIn("12345", actual.loc[0, "text"])

    def test_source_extraction_real_image_count_and_document_budget(self):
        post = {
            "urn": "P", "postedDateTimestamp": 1700000000000,
            "text": "", "contentType": "share", "totalReactionCount": 1234567,
            "images": [{}, {}], "image": [{}, {}, {}, {}, {}],
            "mentions": [{"code": "PERSON_4"}], "companyMentions": [{"code": "ORG_8"}],
            "article": {"title": "Own article", "subtitle": "NEVER INCLUDE SUBTITLE"},
            "document": {"title": "Own document", "totalPageCount": 7,
                         "transcript": ["a" * DOCUMENT_MAX_CHARS, "TRUNCATED TAIL"]},
            "resharedPost": {
                "postedDateTimestamp": 1699999999000, "text": "Shared ⟨PERSON_5⟩",
                "company": {"code": "ORG_3"}, "totalReactionCount": 99887766,
                "images": [{}], "article": {"title": "Shared article"},
                "document": {"title": "Shared document", "totalPageCount": 2,
                             "transcript": ["page one", "page two"]},
            },
        }
        row = extract_post("AUTHOR_A", post)
        self.assertEqual(row["text"], "")
        self.assertEqual(row["shared_text"], "Shared ⟨PERSON⟩")
        self.assertEqual(row["n_images"], 2)
        self.assertEqual(row["shared_n_images"], 1)
        self.assertEqual(row["n_document_pages"], 7)
        self.assertEqual(row["shared_document_pages"], 2)
        self.assertEqual(len(row["document_text"]), DOCUMENT_MAX_CHARS)
        self.assertTrue(row["document_truncated"])
        self.assertEqual(row["shared_document_text"], "page one\npage two")
        self.assertEqual(row["shared_from_company"], 1)
        self.assertEqual(row["n_person_mentions"], 1)
        self.assertEqual(row["n_company_mentions"], 1)
        for value in ["[shared_text]", "Own article", "Own document", "Shared article", "Shared document"]:
            self.assertIn(value, row["content_text"])
        for value in ["TRUNCATED TAIL", "NEVER INCLUDE SUBTITLE", "99887766", "1234567", "PERSON_5"]:
            self.assertNotIn(value, row["content_text"])

    def test_future_dated_shared_source_is_fully_excluded(self):
        row = extract_post("AUTHOR_A", {
            "urn": "P", "postedDateTimestamp": 1700000000000, "text": "own text",
            "resharedPost": {"postedDateTimestamp": 1700000000001,
                             "text": "future ⟨PERSON_1⟩", "images": [{}, {}],
                             "company": {"code": "ORG_1"},
                             "mentions": [{"code": "PERSON_2"}],
                             "article": {"title": "future article"},
                             "document": {"title": "future document", "transcript": ["future transcript"]}},
        })
        self.assertTrue(row["future_shared_excluded"])
        self.assertEqual(row["has_shared"], 0)
        self.assertEqual(row["shared_n_images"], 0)
        self.assertEqual(row["n_person_mentions"], 0)
        self.assertEqual(row["person_codes"], ())
        self.assertNotIn("future", row["content_text"])
        self.assertEqual(row["content_text"], "[text]\nown text")

    def test_shared_publishers_count_for_novelty_without_text_mentions(self):
        rows = []
        for i in range(2):
            rows.append(extract_post("AUTHOR_A", {
                "urn": f"P{i}", "postedDateTimestamp": 1700000000000 + i * 86400000,
                "text": "own text", "resharedPost": {
                    "postedDateTimestamp": 1699999999000,
                    "company": {"code": "ORG_7"}, "author": {"code": "PERSON_8"},
                },
            }, "2025-01-01T00:00:00Z"))
        history = pd.DataFrame(rows)
        history["crawled_at"] = pd.to_datetime(history.crawled_at, utc=True)
        built = build_features(history)
        self.assertEqual(built.org_new_count.tolist(), [1, 0])
        self.assertEqual(built.person_new_count.tolist(), [1, 0])
        self.assertEqual(built.org_seen_fraction.tolist(), [0, 1])
        self.assertEqual(built.n_company_mentions.tolist(), [0, 0])
        self.assertNotIn("ORG_7", " ".join(built.content_text))

    def test_ambiguous_history_dates_fail_before_feature_generation(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                history = history_fixture()
                history.loc[1, "posted_at"] = pd.NaT if missing else history.loc[0, "posted_at"]
                with self.assertRaisesRegex(ValueError, "publication timestamps|tied timestamps"):
                    build_features(history)


class ExperimentReleaseContractTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sma-feature-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = make_release(Path(self.temporary.name) / "release")

    def test_loader_masks_test_answers_keeps_history_order_and_fingerprints(self):
        labelled, history, groups, report = load_experiment_data(self.root)
        original = pd.read_csv(self.root / "labels.csv")
        pd.testing.assert_frame_equal(labelled[["author", "post"]], original[["author", "post"]])
        self.assertTrue(labelled.loc[labelled.split == "test", "label"].isna().all())
        self.assertEqual(set(labelled.loc[labelled.split == "train", "label"]), {0, 1, 2})
        self.assertEqual(set(labelled), {"author", "post", "split", "label", "text", "content_text", *NUMERIC, *CATEGORICAL})
        self.assertFalse({"reactions", "log_reactions", "resid", "class"} & set(labelled))
        self.assertEqual(len(history), 750)
        self.assertEqual(int(history.level.isna().sum()), 150)
        pd.testing.assert_frame_equal(history, history.sort_values(["author", "posted_at"]).reset_index(drop=True))
        self.assertEqual(groups, FEATURE_GROUPS)
        json.dumps(report, allow_nan=False)
        _, _, _, repeated = load_experiment_data(self.root)
        self.assertEqual(report, repeated)
        (self.root / "media").mkdir()
        (self.root / "media" / "ignored.txt").write_text("not part of input fingerprint")
        _, _, _, with_media = load_experiment_data(self.root)
        self.assertEqual(report["input_sha256"], with_media["input_sha256"])
        profile = self.root / "authors" / "AUTHOR_0000.json"
        content = json.loads(profile.read_text())
        content["headline"] = "Unused profile field"
        profile.write_text(json.dumps(content))
        changed, _, _, changed_report = load_experiment_data(self.root)
        self.assertNotEqual(report["input_sha256"], changed_report["input_sha256"])
        pd.testing.assert_frame_equal(labelled, changed)

    def test_loader_refuses_label_or_cutoff_mismatch(self):
        split_path = self.root / "split.json"
        split = json.loads(split_path.read_text())
        split["cutoffs_on_log_residual"]["low"] += 0.1
        split_path.write_text(json.dumps(split))
        with self.assertRaisesRegex(ValueError, "cutoffs differ"):
            load_experiment_data(self.root)
        split["cutoffs_on_log_residual"]["low"] -= 0.1
        split_path.write_text(json.dumps(split))
        labels_path = self.root / "labels.csv"
        labels = pd.read_csv(labels_path)
        labels.loc[0, "label"] = (labels.loc[0, "label"] + 1) % 3
        labels = labels.drop(columns="class")
        labels.to_csv(labels_path, index=False)
        with self.assertRaisesRegex(ValueError, "differ from recomputed"):
            load_experiment_data(self.root)


if __name__ == "__main__":
    unittest.main()
