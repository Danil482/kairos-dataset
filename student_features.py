"""Features for Maria's experiments; only strictly earlier posts enter history.

Counts of previous reactions still come from the collection-time snapshot.
This is a retrospective benchmark, not a reconstruction of live engagement.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

IDENTIFIER = re.compile(r"⟨(PERSON|ORG|AUTHOR)_\d+⟩")
TOKEN = re.compile(r"\w{2,}", re.UNICODE)
CATEGORY = ["content_type"]
SNAPSHOT = ["age_days", "log_followers"]


def normalize_text(text: str) -> str:
    """Keep the kind of placeholder, remove its potentially identifying code."""
    return IDENTIFIER.sub(r"⟨\1⟩", str(text or ""))


def build_features(posts, followers=None, low=-0.549306, high=1.157453):
    if posts.duplicated(["author", "post"]).any():
        raise ValueError("Duplicate author/post keys")
    if posts.posted_at.isna().any():
        raise ValueError("Missing publication timestamps: inspect before modelling")
    df = posts.sort_values(["author", "posted_at", "post"], kind="stable").copy()
    df = df.reset_index(drop=True)
    df["text"] = df.text.map(normalize_text)
    text = df.text
    df["content_type"] = df.content_type.fillna("unknown").astype(str)
    df["log_chars"] = np.log1p(text.str.len())
    df["log_words"] = np.log1p(text.str.split().str.len())
    df["lines"] = text.str.count("\n") + 1
    df["hashtags"] = text.str.count(r"#\w+")
    df["urls"] = text.str.count(r"https?://|⟨URL⟩")
    df["mentions"] = text.str.count(r"@\w+|⟨(?:PERSON|AUTHOR|HANDLE)[^⟩]*⟩")
    df["questions"] = text.str.count(r"\?")
    df["exclamations"] = text.str.count("!")
    df["textless"] = (text.str.len() == 0).astype(int)
    df["weekday"] = df.posted_at.dt.day_of_week
    df["hour_sin"] = np.sin(2 * np.pi * df.posted_at.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.posted_at.dt.hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * df.posted_at.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.posted_at.dt.month / 12)
    content = [c for c in df if c not in posts.columns]
    history_rows = []
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    for _, group in df.groupby("author", sort=False):
        # total_seconds is explicit about units even with pandas us/ms storage.
        seconds = (group.posted_at - epoch).dt.total_seconds().to_numpy()
        logs = np.log1p(group.reactions.clip(lower=0).to_numpy(dtype=float))
        lengths = group.log_chars.to_numpy()
        types = group.content_type.to_numpy()
        tokens = [set(TOKEN.findall(t.lower())) for t in group.text]
        for i in range(len(group)):
            end = int(np.searchsorted(seconds, seconds[i], side="left"))
            past = logs[:end]
            row = {"hist_n": end}
            if end:
                level = float(np.median(past))
                row.update(
                    level=level, hist_mean=float(past.mean()), hist_std=float(past.std()),
                    hist_iqr=float(np.quantile(past, .75) - np.quantile(past, .25)),
                    hist_zero=float(np.mean(past == 0)),
                    recent_mean_delta=float(past[-5:].mean() - level),
                    recent_std=float(past[-10:].std()),
                    history_span_days=(seconds[i] - seconds[0]) / 86400,
                    days_since_prev=(seconds[i] - seconds[end - 1]) / 86400,
                    length_delta=float(lengths[i] - np.mean(lengths[:end])),
                    type_share=float(np.mean(types[:end] == types[i])),
                )
                for days in (7, 30, 90):
                    start = np.searchsorted(seconds[:end], seconds[i] - days * 86400, side="left")
                    row[f"posts_{days}d"] = int(end - start)
                for suffix, values in (("all", past), ("recent", past[-10:]),
                                       ("type", past[types[:end] == types[i]])):
                    # Shrink empirical frequencies toward 20/60/20 for short histories.
                    below = np.sum(values < level + low)
                    above = np.sum(values > level + high)
                    row[f"cdf_below_{suffix}"] = (below + 1) / (len(values) + 5)
                    row[f"cdf_above_{suffix}"] = (above + 1) / (len(values) + 5)
                similarities = [len(tokens[i] & previous) / max(1, len(tokens[i] | previous))
                                for previous in tokens[max(0, end - 5):end]]
                row["recent_text_similarity"] = max(similarities, default=0.)
            history_rows.append(row)
    history = pd.DataFrame(history_rows, index=df.index)
    for col in history:
        df[col] = history[col]
    df["age_days"] = (df.crawled_at - df.posted_at).dt.total_seconds() / 86400
    if followers is None:
        df["log_followers"] = np.nan
    else:
        df["log_followers"] = np.log1p(df.author.map(followers).clip(lower=0))
    numeric = content + list(history.columns)
    # Retain only an explicit whitelist. Current targets cannot enter model inputs.
    output = df[["author", "post", "text"] + CATEGORY + numeric + SNAPSHOT].copy()
    return output, numeric
