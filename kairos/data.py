"""Read the release from disk.

The core archive unpacks into one folder, `data/` by default:

    authors/AUTHOR_xxxx.json   one profile per author
    posts/AUTHOR_xxxx.json     that author's feed as it was served
    media/                     media files, once each, and media/manifest.jsonl
    labels.csv                 author, post, split, label, class
    split.json                 train and test authors, and the class cutoffs
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _release(data_dir: Path | str) -> Path:
    data_dir = Path(data_dir)
    if not (data_dir / "posts").is_dir() or not (data_dir / "labels.csv").is_file():
        raise SystemExit(
            f"No release in {data_dir}. Request the password and download the core "
            "archive at https://kairos-labs.ai/dataset/, then unpack it there:\n"
            f"    7z x lab1_core.7z.001 -o{data_dir}"
        )
    return data_dir


def load_posts(data_dir: Path | str = DATA_DIR, own_only: bool = True) -> pd.DataFrame:
    """One row per post, in file order.

    A feed holds the author's own posts and the posts they reposted. A plain
    repost arrives as the original post, with the original's author and the
    original's counters, so it says nothing about how this author's posts do.
    `own_only` keeps the posts whose `author.code` is the feed's owner, which is
    the population the labels are defined on.
    """
    data_dir = _release(data_dir)
    rows = []
    for f in tqdm(
        sorted((data_dir / "posts").glob("AUTHOR_*.json")), desc="posts", unit="file"
    ):
        feed = json.loads(f.read_text(encoding="utf-8"))
        owner = feed["author"]
        for post in feed["posts"]:
            a = post.get("author")
            own = isinstance(a, dict) and a.get("code") == owner
            if own_only and not own:
                continue
            rows.append(
                {
                    "author": owner,
                    "post": post["urn"],
                    "own": own,
                    # `postedAt` is LinkedIn's relative display string ("2yr", "1mo");
                    # the timestamp is `postedDateTimestamp`, in ms since the epoch.
                    "posted_at": post.get("postedDateTimestamp"),
                    "crawled_at": feed.get("crawledAt"),
                    # The provider omits a counter at zero, so an absent count is 0.
                    "reactions": int(post.get("totalReactionCount") or 0),
                    "text": post.get("text") or "",
                    "content_type": post.get("contentType") or "unknown",
                }
            )
    df = pd.DataFrame(rows)
    df["posted_at"] = pd.to_datetime(df["posted_at"], unit="ms", utc=True)
    # The release mixes ISO timestamps with and without fractional seconds.
    df["crawled_at"] = pd.to_datetime(
        df["crawled_at"], format="ISO8601", utc=True, errors="coerce"
    )
    return df


def load_followers(data_dir: Path | str = DATA_DIR) -> pd.Series:
    """Follower count per author, read in 2026: today's audience, not the post's."""
    data_dir = _release(data_dir)
    counts = {}
    for f in sorted((data_dir / "authors").glob("AUTHOR_*.json")):
        profile = json.loads(f.read_text(encoding="utf-8"))
        counts[profile.get("author", f.stem)] = profile.get("followers")
    return pd.to_numeric(pd.Series(counts, name="followers"), errors="coerce")


def load_labels(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """author, post, split ("train" / "test"), label (0 / 1 / 2), class."""
    return pd.read_csv(_release(data_dir) / "labels.csv")


def load_split(data_dir: Path | str = DATA_DIR) -> dict:
    return json.loads((_release(data_dir) / "split.json").read_text(encoding="utf-8"))
