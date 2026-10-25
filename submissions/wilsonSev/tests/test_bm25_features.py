import unittest
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from kairos.bm25_features import author_bm25_features, bm25_scores


class BM25Tests(unittest.TestCase):
    def setUp(self):
        self.posts = pd.DataFrame(
            {
                "author": ["A"] * 7 + ["B", "B"],
                "post": ["a0", "a1", "a2", "a3", "a4", "a5", "missing", "b0", "b1"],
                "posted_at": pd.to_datetime(
                    [
                        "2026-01-01",
                        "2026-01-02",
                        "2026-01-02",
                        "2026-01-03",
                        "2026-01-04",
                        "2026-01-05",
                        None,
                        "2026-01-01",
                        "2026-01-02",
                    ]
                ),
                "text": [
                    "python data",
                    "travel",
                    "python python",
                    "python",
                    "travel python",
                    None,
                    "python",
                    "python",
                    "python",
                ],
                "reactions": [5, 100, 1000, 20, 80, 10, 99999, 5000, 4],
            }
        )

    def test_hand_calculation(self):
        docs = ["python " + "other " * 99, "python " * 3 + "other " * 97] + [
            "other " * 100
        ] * 8
        scores = bm25_scores("python", docs)
        idf = np.log(4.4)
        self.assertAlmostEqual(scores[0], idf)
        self.assertAlmostEqual(scores[1], idf * 6.6 / 4.2)
        np.testing.assert_array_equal(scores[2:], 0)
        np.testing.assert_array_equal(bm25_scores("python python", docs), scores)
        self.assertEqual(bm25_scores("python", [""])[0], 0)

    def test_matches_naive_formula_top_k_and_relative_outcome(self):
        posts = self.posts.sample(frac=1, random_state=3)
        actual = author_bm25_features(posts, top_k=2)
        self.assertEqual(actual.post.tolist(), posts.post.tolist())
        tokenize = CountVectorizer().build_analyzer()
        for (_, current), (_, result) in zip(posts.iterrows(), actual.iterrows()):
            past = posts[
                (posts.author == current.author) & (posts.posted_at < current.posted_at)
            ]
            past = past.sort_values(["posted_at", "post"])
            documents = [Counter(tokenize(s or "")) for s in past.text.fillna("")]
            lengths = [sum(d.values()) for d in documents]
            scores = np.zeros(len(past))
            if sum(lengths):
                for word in set(
                    tokenize("" if pd.isna(current.text) else current.text)
                ):
                    df = sum(word in d for d in documents)
                    idf = np.log(1 + (len(past) - df + 0.5) / (df + 0.5))
                    for j, d in enumerate(documents):
                        tf = d[word]
                        scores[j] += (
                            idf
                            * tf
                            * 2.2
                            / (tf + 1.2 * (0.25 + 0.75 * lengths[j] / np.mean(lengths)))
                        )
            selected = sorted(
                (j for j in range(len(past)) if scores[j] > 0), key=lambda j: -scores[j]
            )[:2]
            self.assertEqual(result.bm25_top_posts, past.iloc[selected].post.tolist())
            np.testing.assert_allclose(result.bm25_top_scores, scores[selected])
            expected = 0
            if selected:
                expected = (
                    np.log1p(past.iloc[selected].reactions).mean()
                    - np.log1p(past.reactions).median()
                )
            self.assertAlmostEqual(result.bm25_history_delta, expected)
            self.assertEqual(result.bm25_match_count, len(selected))

    def test_future_current_reactions_and_other_author_cannot_leak(self):
        before = author_bm25_features(self.posts)
        modified = self.posts.copy()
        modified.loc[
            modified.post.isin(["a1", "a2", "a3", "a4", "a5"]), "reactions"
        ] = 999999
        modified.loc[modified.author == "B", "text"] = "new arbitrary text"
        after = author_bm25_features(modified)
        # a1 and a2 are simultaneous: neither reads the other's reactions.
        pd.testing.assert_frame_equal(before.iloc[:3], after.iloc[:3])
        future = self.posts.iloc[[0]].assign(
            post="future",
            posted_at=pd.Timestamp("2027-01-01"),
            text="python " * 100,
            reactions=999999,
        )
        appended = author_bm25_features(
            pd.concat([self.posts, future], ignore_index=True)
        )
        pd.testing.assert_frame_equal(before, appended.iloc[:-1].reset_index(drop=True))

    def test_ties_empty_queries_and_missing_dates(self):
        posts = self.posts.iloc[:5].copy()
        posts["text"] = "python"
        result = author_bm25_features(
            posts.sample(frac=1, random_state=2), top_k=2
        ).set_index("post")
        self.assertEqual(result.loc["a4", "bm25_top_posts"], ["a0", "a1"])
        full = author_bm25_features(self.posts).set_index("post")
        self.assertEqual(full.loc["a5", "bm25_match_count"], 0)
        self.assertEqual(full.loc["missing", "bm25_match_count"], 0)
        self.assertEqual(full.loc["b1", "bm25_top_posts"], ["b0"])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            author_bm25_features(self.posts, top_k=0)
        with self.assertRaises(ValueError):
            author_bm25_features(pd.concat([self.posts, self.posts.iloc[:1]]))
        with self.assertRaises(ValueError):
            bm25_scores("x", ["x"], b=2)


if __name__ == "__main__":
    unittest.main()
