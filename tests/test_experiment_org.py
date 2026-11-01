"""Causal and partition contracts for target-free organisation graph features."""

import math
import unittest

import numpy as np
import pandas as pd

from kairos.experiment_org import FEATURE_COLUMNS, organization_features


def frame(rows):
    result = pd.DataFrame(rows, columns=["author", "post", "day", "org_codes"])
    result["posted_at"] = pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(result.pop("day"), unit="D")
    return result


def fixture():
    return frame([
        ("A", "a0", 1, ("ORG_A", "ORG_Y")),
        ("B", "b0", 2, ("ORG_A", "ORG_X")),
        ("B", "b1", 3, ("ORG_X", "ORG_X")),
        ("C", "c0", 3, ("ORG_X", "ORG_Y")),
        ("D", "d0", 4, ("ORG_A", "ORG_Z")),
        ("A", "query", 5, ("ORG_X", "ORG_Y", "ORG_Z")),
    ])


class OrganizationFeatureTest(unittest.TestCase):
    def test_hand_computed_counts_deduplicate_authors_and_current_orgs(self):
        history = fixture()
        result = organization_features(history, history.iloc[[-1]], {"A", "B", "C"}).iloc[0]
        self.assertEqual(result.org_net_n_orgs, 3)
        self.assertAlmostEqual(result.org_net_known_fraction, 2 / 3)
        self.assertAlmostEqual(result.org_net_log_authors_mean, (math.log(3) + math.log(2)) / 3)
        self.assertAlmostEqual(result.org_net_log_authors_max, math.log(3))
        self.assertAlmostEqual(result.org_net_cooccurrence_fraction, 1 / 3)
        self.assertEqual(result.org_net_no_orgs, 0)
        self.assertEqual(result.org_net_all_unseen, 0)
        self.assertEqual(result.org_net_no_own_org_history, 0)

    def test_same_timestamp_and_future_bank_posts_do_not_enter_any_feature(self):
        history = fixture()
        query = history.iloc[[-1]]
        expected = organization_features(history, query, {"A", "B", "C", "E"})
        extra = frame([
            ("E", "same", 5, ("ORG_A", "ORG_Y", "ORG_Z")),
            ("E", "future", 6, ("ORG_A", "ORG_Y", "ORG_Z")),
            ("A", "own-same", 5, ("ORG_X",)),
            ("A", "own-future", 7, ("ORG_X", "ORG_Z")),
        ])
        actual = organization_features(pd.concat([extra, history], ignore_index=True), query,
                                       {"A", "B", "C", "E"})
        pd.testing.assert_frame_equal(expected, actual)

    def test_query_owner_never_supplies_global_popularity_or_cooccurrence(self):
        history = fixture()
        query = history.iloc[[-1]]
        with_owner = organization_features(history, query, {"A", "B", "C"})
        without_owner = organization_features(history, query, {"B", "C"})
        pd.testing.assert_frame_equal(with_owner, without_owner)
        own_only = organization_features(history, query, {"A"}).iloc[0]
        self.assertEqual(own_only.org_net_known_fraction, 0)
        self.assertEqual(own_only.org_net_cooccurrence_fraction, 0)
        self.assertEqual(own_only.org_net_all_unseen, 1)

    def test_nonbank_content_and_all_targets_are_irrelevant(self):
        history = fixture()
        query = history.iloc[[-1]]
        expected = organization_features(history, query, {"B", "C"})
        changed = history.copy()
        changed.at[4, "org_codes"] = ("ORG_A", "ORG_Y", "ORG_Z")
        changed["label"] = [2, 0, 1, 0, 2, 1]
        changed["reactions"] = np.arange(len(changed)) * 1e12
        changed["resid"] = -987654321
        actual = organization_features(changed, query, {"B", "C"})
        pd.testing.assert_frame_equal(expected, actual)
        self.assertEqual(list(actual), ["author", "post"] + FEATURE_COLUMNS)

    def test_heldout_owner_can_use_own_past_but_not_current_orgs_as_history(self):
        history = frame([
            ("A", "past", 1, ("ORG_A",)),
            ("B", "bank", 2, ("ORG_A", "ORG_B", "ORG_C")),
            ("A", "query", 3, ("ORG_B",)),
            ("D", "first", 3, ("ORG_B", "ORG_C")),
        ])
        result = organization_features(history, history.iloc[[2, 3]], {"B"})
        self.assertEqual(result.iloc[0].org_net_cooccurrence_fraction, 1)
        self.assertEqual(result.iloc[1].org_net_cooccurrence_fraction, 0)
        self.assertEqual(result.iloc[1].org_net_no_own_org_history, 1)

    def test_shared_pair_excludes_owner_but_keeps_another_source_author(self):
        history = frame([
            ("A", "own", 1, ("ORG_A", "ORG_B")),
            ("B", "other", 2, ("ORG_A", "ORG_B")),
            ("A", "query", 3, ("ORG_B",)),
        ])
        result = organization_features(history, history.iloc[[-1]], {"A", "B"})
        self.assertEqual(result.iloc[0].org_net_cooccurrence_fraction, 1)

    def test_absent_orgs_and_unknown_orgs_have_distinct_flags(self):
        history = frame([("A", "empty", 1, ()), ("A", "unknown", 2, ("ORG_Z",))])
        result = organization_features(history, history, set())
        self.assertEqual(result.org_net_no_orgs.tolist(), [1, 0])
        self.assertEqual(result.org_net_all_unseen.tolist(), [0, 1])
        self.assertTrue(np.isfinite(result[FEATURE_COLUMNS].to_numpy()).all())

    def test_unsorted_history_and_repeated_queries_preserve_query_index_and_order(self):
        history = fixture()
        queries = history.iloc[[5, 2, 5]][["author", "post"]].copy()
        queries.index = pd.Index([91, 7, 91], name="original")
        expected = organization_features(history, queries, {"A", "B", "C"})
        actual = organization_features(history.sample(frac=1, random_state=3), queries,
                                       {"A", "B", "C"})
        pd.testing.assert_frame_equal(expected, actual)
        pd.testing.assert_frame_equal(actual[["author", "post"]], queries)

    def test_empty_query_batch_has_the_complete_schema(self):
        history = fixture()
        result = organization_features(history, history.iloc[:0], {"B"})
        self.assertEqual(len(result), 0)
        self.assertEqual(list(result), ["author", "post"] + FEATURE_COLUMNS)

    def test_sweep_matches_direct_definition_with_tied_times_and_changing_banks(self):
        rng = np.random.default_rng(42)
        rows = [(str(rng.choice(["A", "B", "C", "D"])), str(i), int(rng.integers(1, 9)),
                 tuple(rng.choice(["ORG_1", "ORG_2", "ORG_3", "ORG_4"],
                                  size=int(rng.integers(0, 4)), replace=False)))
                for i in range(40)]
        history = frame(rows)
        for bank in [set(), {"B"}, {"A", "B", "C", "D"}]:
            expected = []
            for query in history.itertuples():
                past = history[history.posted_at < query.posted_at]
                own = set().union(*past.loc[past.author.eq(query.author), "org_codes"])
                other = past[past.author.isin(bank) & past.author.ne(query.author)]
                counts, connected = [], 0
                for code in query.org_codes:
                    matching = other.loc[other.org_codes.map(lambda source: code in source).astype(bool)]
                    counts.append(matching.author.nunique())
                    connected += any((set(source) - {code}) & own for source in matching.org_codes)
                n = len(query.org_codes)
                known = sum(count > 0 for count in counts)
                popularity = [math.log1p(count) for count in counts]
                expected.append([n, known / max(1, n), sum(popularity) / max(1, n),
                                 max(popularity, default=0), connected / max(1, n),
                                 int(n == 0), int(n > 0 and known == 0), int(not own)])
            actual = organization_features(history, history, bank)
            np.testing.assert_allclose(actual[FEATURE_COLUMNS], expected, rtol=0, atol=1e-14)

    def test_missing_duplicate_and_unknown_keys_or_times_fail_explicitly(self):
        history = fixture()
        queries = history.iloc[[-1]]
        for bad_history, fragment in [
            (pd.concat([history, history.iloc[[0]]]), "duplicate"),
            (history.assign(posted_at=pd.NaT), "timestamps"),
            (history.assign(author=None), "missing"),
        ]:
            with self.subTest(fragment=fragment), self.assertRaisesRegex(ValueError, fragment):
                organization_features(bad_history, queries, {"B"})
        with self.assertRaisesRegex(ValueError, "absent"):
            organization_features(history, queries.assign(post="not-present"), {"B"})
        with self.assertRaisesRegex(ValueError, "ORG_"):
            organization_features(history.assign(org_codes="ORG_A"), queries, {"B"})


if __name__ == "__main__":
    unittest.main()
