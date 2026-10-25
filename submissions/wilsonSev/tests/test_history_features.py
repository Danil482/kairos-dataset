"""Run with: python -m unittest discover -s tests -p 'test_history_features.py'."""

import unittest
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from kairos.history_features import author_text_similarity


class HistoryFeatureTests(unittest.TestCase):
    def setUp(self):
        self.vectorizer = TfidfVectorizer().fit(["python data", "marketing job"])
        self.posts = (
            pd.DataFrame(
                {
                    "author": ["A", "A", "A", "A", "B", "B", "A", "A"],
                    "post": ["a1", "a2", "a3", "a4", "b1", "b2", "missing", "empty"],
                    "posted_at": pd.to_datetime(
                        [
                            "2026-01-01",
                            "2026-01-02",
                            "2026-01-02",
                            "2026-01-03",
                            "2026-01-01",
                            "2026-01-02",
                            None,
                            "2026-01-04",
                        ]
                    ),
                    "text": [
                        "python",
                        "python",
                        "marketing",
                        "marketing",
                        "marketing",
                        "marketing",
                        "python",
                        "unseenword",
                    ],
                }
            )
            .sample(frac=1, random_state=7)
            .set_index("post", drop=False)
        )

    def test_matches_explicit_strict_history_and_preserves_order(self):
        result = author_text_similarity(self.posts, self.vectorizer)
        self.assertEqual(result["post"].tolist(), self.posts["post"].tolist())
        for (_, post), (_, actual) in zip(self.posts.iterrows(), result.iterrows()):
            past = self.posts[
                (self.posts.author == post.author)
                & (self.posts.posted_at < post.posted_at)
            ]
            expected = 0.0
            if len(past):
                mean = np.asarray(self.vectorizer.transform(past.text).mean(axis=0))
                expected = cosine_similarity(
                    self.vectorizer.transform([post.text]), mean
                )[0, 0]
            self.assertAlmostEqual(actual.history_text_similarity, expected)
            self.assertEqual(actual.text_history_n, len(past))
        by_post = result.set_index("post")
        self.assertEqual(by_post.loc["a3", "history_text_similarity"], 0)
        self.assertEqual(by_post.loc["a4", "text_history_n"], 3)
        self.assertEqual(by_post.loc["b1", "history_text_similarity"], 0)
        self.assertEqual(by_post.loc["b2", "history_text_similarity"], 1)
        self.assertEqual(by_post.loc["missing", "history_text_similarity"], 0)
        self.assertEqual(by_post.loc["empty", "history_text_similarity"], 0)

    def test_future_posts_do_not_change_existing_features_or_idf(self):
        before_idf = self.vectorizer.idf_.copy()
        before = author_text_similarity(self.posts, self.vectorizer)
        future = pd.DataFrame(
            [
                dict(
                    author="A",
                    post="future",
                    posted_at=pd.Timestamp("2027-01-01"),
                    text="marketing",
                )
            ]
        )
        after = author_text_similarity(
            pd.concat([self.posts.reset_index(drop=True), future], ignore_index=True),
            self.vectorizer,
        )
        pd.testing.assert_frame_equal(before, after.iloc[:-1].reset_index(drop=True))
        np.testing.assert_array_equal(before_idf, self.vectorizer.idf_)

    def test_duplicate_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            author_text_similarity(
                pd.concat([self.posts, self.posts.iloc[:1]]), self.vectorizer
            )

    def test_zero_history_profile(self):
        posts = self.posts.loc[["a1", "a2"]].copy()
        posts.loc["a1", "text"] = ""
        result = author_text_similarity(posts, self.vectorizer)
        self.assertEqual(result.iloc[1].history_text_similarity, 0)


if __name__ == "__main__":
    unittest.main()
