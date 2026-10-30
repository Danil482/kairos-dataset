"""Formatting features: Unicode emoji, punctuation, structure, and author history."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from kairos import features
from kairos.label import author_level


class FormattingValuesTests(unittest.TestCase):
    def test_compound_emoji_are_whole_items_and_leading_whitespace_is_ignored(self):
        actual = features._formatting_values(" \n👩🏽‍💻 👩🏽‍💻 🇷🇺 1️⃣")
        self.assertEqual(actual["n_emoji"], 4)
        self.assertEqual(actual["n_unique_emoji"], 3)
        self.assertEqual(actual["emoji_at_start"], 1)
        self.assertEqual(features._formatting_values("Hello 🚀")["emoji_at_start"], 0)
        self.assertEqual(features._formatting_values("123 # *")["n_emoji"], 0)

    def test_punctuation_runs_and_case_use_the_specified_denominators(self):
        actual = features._formatting_values("AA aa ББ бб A 中文 123!!!?? ?! ! ?")
        self.assertEqual(actual["n_exclamations"], 5)
        self.assertEqual(actual["n_questions"], 4)
        self.assertEqual(actual["n_repeated_punct"], 2)
        self.assertAlmostEqual(actual["uppercase_ratio"], 5 / 9)
        self.assertEqual(actual["n_uppercase_words"], 2)

    def test_combining_marks_do_not_make_a_single_letter_an_uppercase_word(self):
        actual = features._formatting_values("ÉÉ e\u0301e\u0301 A\u0301 中文")
        self.assertAlmostEqual(actual["uppercase_ratio"], 3 / 5)
        self.assertEqual(actual["n_uppercase_words"], 1)

    def test_paragraphs_list_markers_and_windows_line_endings(self):
        middle = "- First\n2) Second\n(3) Third\n• Fourth\n✅ Fifth"
        text = " Intro \r\n\r\n" + middle.replace("\n", "\r\n") + "\r\n\r\nTail"
        actual = features._formatting_values(text)
        self.assertEqual(actual["n_paragraphs"], 3)
        self.assertEqual(actual["first_paragraph_length"], 5)
        self.assertEqual(actual["max_paragraph_length"], len(middle))
        self.assertEqual(actual["n_list_items"], 5)
        self.assertEqual(features._formatting_values("First\nSecond")["n_paragraphs"], 1)
        self.assertEqual(
            features._formatting_values("---\n***\nplain - text\n3.14 value")["n_list_items"],
            0,
        )

    def test_empty_text_and_text_without_cased_letters_have_finite_zero_counts(self):
        actual = features._formatting_values(" \r\n\t\n ")
        self.assertTrue(all(value == 0 for value in actual.values()))
        self.assertEqual(features._formatting_values("中文 123 🚀")["uppercase_ratio"], 0)


class FormattingHistoryTests(unittest.TestCase):
    def test_strict_history_preserves_rows_and_excludes_other_authors_and_missing_dates(self):
        posts = pd.DataFrame(
            {
                "author": ["a", "a", "a", "b", "a", "a", "a"],
                "posted_at": pd.to_datetime(
                    [
                        "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-01",
                        "2025-01-03", "2025-01-04", None,
                    ],
                    utc=True,
                ),
                "text": [
                    "aa", "AA 🚀\n\nBB", "cc 🚀🚀", "🚀🚀🚀 XXX\n\nY\n\nZ",
                    "DD 🚀🚀🚀", "zz", "🚀🚀🚀🚀 AA",
                ],
            },
            index=[50, 7, 91, 3, 22, 15, 8],
        ).iloc[[2, 3, 0, 6, 4, 5, 1]]
        actual = features._formatting(posts)
        pd.testing.assert_index_equal(actual.index, posts.index)
        columns = ["emoji_vs_history", "uppercase_vs_history", "paragraphs_vs_history"]
        np.testing.assert_allclose(actual.loc[91, columns], [1.5, -0.5, -0.5])
        np.testing.assert_allclose(actual.loc[22, columns], [2.5, 0.5, -0.5])
        np.testing.assert_allclose(actual.loc[7, columns], [1, 1, 1])
        np.testing.assert_allclose(actual.loc[15, columns], [-1.5, -0.5, -0.25])
        np.testing.assert_array_equal(actual.loc[[50, 3, 8], columns], np.zeros((3, 3)))
        self.assertEqual(actual.loc[8, "n_emoji"], 4)

        changed = posts.copy()
        changed.loc[15, "text"] = "🚀" * 100
        repeated = features._formatting(changed)
        pd.testing.assert_frame_equal(actual.drop(index=15), repeated.drop(index=15))

    def test_only_last_twenty_posts_contribute_to_the_mean(self):
        posts = pd.DataFrame(
            {
                "author": ["a"] * 22,
                "posted_at": pd.date_range("2025-01-01", periods=22, tz="UTC"),
                "text": ["🚀🚀 AA\n\nBB"] + ["aa"] * 20 + ["🚀 aa"],
            }
        )
        with patch.object(features, "FORMATTING_HISTORY", 20):
            with patch.object(features, "N_LAST", 1):
                actual = features._formatting(posts)
        np.testing.assert_allclose(actual.iloc[-1, -3:], [1, 0, 0])

    def test_build_exposes_numeric_formatting_features_without_text_encoders(self):
        posts = pd.DataFrame(
            {
                "author": ["a"] * 6,
                "post": [f"p{i}" for i in range(6)],
                "posted_at": pd.date_range("2025-01-01", periods=6, tz="UTC"),
                "crawled_at": pd.Timestamp("2026-01-01", tz="UTC"),
                "reactions": [1, 2, 3, 4, 5, 6],
                "content_type": ["text"] * 6,
                "text": ["aa", "AA 🚀", "first\n\nsecond", "!?!", "", "BB"],
            },
            index=[9, 4, 20, 8, 1, 32],
        )
        levelled = author_level(posts)
        with patch.object(features, "_e5_embeddings", side_effect=AssertionError):
            with patch.object(features, "TfidfVectorizer", side_effect=AssertionError):
                table = features.build(levelled, pd.Series({"a": 100}))
        self.assertTrue(set(features.FORMATTING_NUM).issubset(table.columns))
        self.assertEqual(len(features.FORMATTING_NUM), 15)
        pd.testing.assert_index_equal(table.index, levelled.index)
        transformer = features.preprocess(features.FORMATTING_NUM)
        matrix = transformer.fit_transform(table)
        self.assertEqual(matrix.shape, (6, 15))
        self.assertTrue(np.isfinite(matrix).all())
        self.assertEqual(set(transformer.named_transformers_) - {"remainder"}, {"num"})


if __name__ == "__main__":
    unittest.main()
