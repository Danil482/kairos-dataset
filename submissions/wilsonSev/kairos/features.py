"""Features of the baseline, in three groups.

content    what the post is: length, hashtags, mentions, links, type, posting time
history    who wrote it: the author's level and spread, rhythm, follower count
artefact   how it was collected: the post's age at crawl and its place in the
           author's run
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from kairos.label import MIN_HISTORY

# The release replaces links, handles and names with placeholders such as ⟨URL⟩,
# ⟨HANDLE⟩ and ⟨PERSON_3⟩, so the counters look for those as well as raw text.
_URL = re.compile(r"https?://|⟨URL⟩")
_MENTION = re.compile(r"@\w+|⟨HANDLE⟩|⟨(?:PERSON|AUTHOR)[^⟩]*⟩")
_HASHTAG = re.compile(r"#(?:\w+|⟨[^⟩]*⟩)")

CONTENT_NUM = [
    "n_chars",
    "n_words",
    "n_lines",
    "n_hashtags",
    "n_mentions",
    "n_urls",
    "has_question",
    "is_textless",
    "hour",
    "weekday",
]
CONTENT_CAT = ["content_type"]
HISTORY_NUM = ["level", "hist_iqr", "days_since_prev", "log_followers"]
ARTEFACT_NUM = ["age_days", "hist_n"]


def build(levelled: pd.DataFrame, followers: pd.Series) -> pd.DataFrame:
    """Every feature for every own post.

    Takes the output of `label.author_level`, sorted by author and time, and
    reads each author's history from earlier posts only, as the level does.
    """
    df = levelled.copy()
    text = df["text"]
    df["n_chars"] = text.str.len()
    df["n_words"] = text.str.split().str.len().fillna(0)
    df["n_lines"] = text.str.count("\n") + 1
    df["n_hashtags"] = text.str.count(_HASHTAG)
    df["n_mentions"] = text.str.count(_MENTION)
    df["n_urls"] = text.str.count(_URL)
    df["has_question"] = text.str.contains("?", regex=False).astype(int)
    df["is_textless"] = (df["n_chars"] == 0).astype(int)
    df["hour"] = df["posted_at"].dt.hour
    df["weekday"] = df["posted_at"].dt.weekday

    # Reactions are one snapshot taken at crawl time, and an older post has had
    # longer to collect them. Age is a property of the collection, not of the
    # post, so it sits in its own group instead of hiding among the content.
    df["age_days"] = (df["crawled_at"] - df["posted_at"]).dt.total_seconds() / 86400
    df["age_days"] = df["age_days"].fillna(df["age_days"].median())

    df["log_followers"] = np.log1p(df["author"].map(followers).fillna(0).clip(lower=0))

    # The spread of the author's past: a level alone cannot say whether this
    # person is steady or erratic, and the cutoffs are global, so an erratic
    # author spends more time in the outer classes.
    past = df.groupby("author")["log_reactions"]
    q75 = past.transform(
        lambda s: s.shift(1).expanding(min_periods=MIN_HISTORY).quantile(0.75)
    )
    q25 = past.transform(
        lambda s: s.shift(1).expanding(min_periods=MIN_HISTORY).quantile(0.25)
    )
    df["hist_iqr"] = (q75 - q25).fillna(0.0)

    previous = df.groupby("author")["posted_at"].shift(1)
    df["days_since_prev"] = (
        (df["posted_at"] - previous).dt.total_seconds() / 86400
    ).fillna(-1.0)
    return df
