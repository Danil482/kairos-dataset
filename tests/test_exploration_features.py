import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.analyze_train import extra_content, lag_features


class ExplorationFeatureTests(unittest.TestCase):
    def sample(self):
        return pd.DataFrame(dict(author=["a"] * 5, post=list("abcde"),
            posted_at=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02", "2024-02-01", "2024-03-01"], utc=True),
            reactions=[0, 2, 7, 10, 30], content_type=["text"] * 5))

    def test_lags_do_not_read_current_or_future_reactions(self):
        posts = self.sample()
        expected = lag_features(posts)
        for i in range(len(posts)):
            modified = posts.copy()
            modified.loc[i:, "reactions"] = 100000
            actual = lag_features(modified)
            pd.testing.assert_series_equal(expected.iloc[i], actual.iloc[i])
        self.assertEqual(expected.iloc[1].last_reaction_delta, 0.)
        self.assertEqual(expected.iloc[2].last_reaction_delta, 0.)

    def test_content_excludes_engagement_counters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data" / "posts").mkdir(parents=True)
            post = dict(urn="post1", author=dict(code="a"), text="hello",
                article=dict(title="article"), document=dict(title="slides", transcript="contents"),
                resharedPost=dict(text="shared content"), totalReactionCount=1, commentsCount=2)
            file = root / "data" / "posts" / "a.json"
            with patch("scripts.analyze_train.ROOT", root):
                file.write_text(json.dumps(dict(posts=[post])), encoding="utf-8")
                expected = extra_content({"a"})
                post.update(totalReactionCount=9999, commentsCount=3333, likeCount=4444)
                post["resharedPost"]["totalReactionCount"] = 5555
                file.write_text(json.dumps(dict(posts=[post])), encoding="utf-8")
                actual = extra_content({"a"})
                pd.testing.assert_frame_equal(expected, actual)
                self.assertEqual(actual.expanded_text.iloc[0], "hello\narticle\nslides\ncontents\nshared content")


if __name__ == "__main__":
    unittest.main()
