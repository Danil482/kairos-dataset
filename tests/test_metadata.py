from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from lab1_metadata import enrich_posts


class MetadataTests(unittest.TestCase):
    def test_crawl_dates_keep_both_iso_precisions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "posts").mkdir()
            (root / "authors").mkdir()
            for i, crawled in enumerate(["2026-01-01T12:00:00.123Z", "2026-01-02T12:00:00Z"]):
                author, post = f"A{i}", f"P{i}"
                (root / "posts" / f"AUTHOR_{i}.json").write_text(json.dumps({
                    "author": author, "crawledAt": crawled,
                    "posts": [{"urn": post, "author": {"code": author}}]}))
                (root / "authors" / f"AUTHOR_{i}.json").write_text(json.dumps({"author": author}))
            raw = pd.DataFrame({"author": ["A0", "A1"], "post": ["P0", "P1"], "crawled_at": [pd.NaT, pd.NaT]})
            result = enrich_posts(raw, root)
            self.assertFalse(result.crawled_at.isna().any())
            self.assertEqual(result.loc[0, "crawled_at"], pd.Timestamp("2026-01-01T12:00:00.123Z"))
            self.assertEqual(result.loc[1, "crawled_at"], pd.Timestamp("2026-01-02T12:00:00Z"))


if __name__ == "__main__":
    unittest.main()
