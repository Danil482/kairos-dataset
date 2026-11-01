from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from kairos import features
from lab1_metadata import POST_NUM, PROFILE_CAT, PROFILE_NUM

BASE_NUMERIC = features.CONTENT_NUM + features.HISTORY_NUM + features.ARTEFACT_NUM
CATEGORICAL = ["content_type"] + PROFILE_CAT
IDENTIFIER = re.compile(r"⟨((?:PERSON|AUTHOR|ORG|POST)(?:_\d+)?)⟩")


def clean_text(text: str) -> str:
    text = IDENTIFIER.sub(lambda m: " " + m.group(1).split("_")[0] + " ", text)
    text = re.sub(r"\b(PERSON|AUTHOR|ORG|POST)_\d+\b", r"\1", text)
    text = re.sub(r"https?://\S+", " URL ", text)
    return text.strip() or "EMPTY_POST"


def build(levelled: pd.DataFrame, followers: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    df = features.build(levelled, followers)
    for column in PROFILE_NUM + POST_NUM:
        if column not in df:
            df[column] = np.nan
    for column in PROFILE_CAT:
        df[column] = df.get(column, pd.Series("unknown", index=df.index)).fillna("unknown").astype(str)

    df["age_days"] = (df["crawled_at"] - df["posted_at"]).dt.total_seconds() / 86400
    added = {}

    def add(name, values):
        added[name] = values

    text = df["text"].fillna("")
    chars = df["n_chars"].clip(lower=1)
    words = df["n_words"].clip(lower=1)
    patterns = {
        "exclamations": r"!", "questions": r"\?", "digits": r"\d",
        "uppercase": r"[A-Z]", "paragraphs": r"\n\s*\n", "bullets": r"(?m)^\s*[-•▪✓✔]",
        "org_mentions": r"⟨ORG[^⟩]*⟩", "person_mentions": r"⟨(?:PERSON|AUTHOR)[^⟩]*⟩",
    }
    for name, pattern in patterns.items():
        count = text.str.count(pattern)
        add("n_" + name, count)
        add("ratio_" + name, count / chars)
    add("word_length", df["n_chars"] / words)
    add("words_per_line", df["n_words"] / df["n_lines"].clip(lower=1))
    add("unique_word_ratio", text.map(lambda s: len(set(s.lower().split()))) / words)
    add("n_symbols", text.map(lambda s: sum(unicodedata.category(c) == "So" for c in s)))
    for col in ["n_chars", "n_words", "n_lines", "n_hashtags", "n_mentions", "n_urls"]:
        add("log_" + col, np.log1p(df[col]))
    for period, values, name in [
        (24, df["hour"], "hour"), (7, df["weekday"], "weekday"),
        (12, df["posted_at"].dt.month, "month"),
    ]:
        add(name + "_sin", np.sin(values * 2 * np.pi / period))
        add(name + "_cos", np.cos(values * 2 * np.pi / period))
    add("year", df["posted_at"].dt.year)
    add("log_age_days", np.log1p(df["age_days"].clip(lower=0)))
    add("log_hist_n", np.log1p(df["hist_n"]))
    add("log_gap", np.log1p(df["days_since_prev"].clip(lower=0)))
    first = df.groupby("author")["posted_at"].transform("min")
    span = (df["posted_at"] - first).dt.total_seconds() / 86400
    add("history_span_days", span)
    add("posting_rate", df["hist_n"] / span.clip(lower=1))


    for source in ["log_reactions", "resid", "n_chars", "n_words", "n_hashtags",
                   "n_urls", "days_since_prev"]:
        grouped = df.groupby("author", sort=False)[source]
        if source in ["log_reactions", "resid"]:
            add(source + "_lag1", grouped.shift(1))
        for window in [3, 5, 10, 20]:
            mean = grouped.transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            add(f"{source}_mean{window}", mean)
            if source in ["log_reactions", "resid"]:
                add(f"{source}_std{window}", grouped.transform(
                    lambda s: s.shift(1).rolling(window, min_periods=2).std()))
            if source == "log_reactions":
                add(f"recent_level_delta{window}", mean - df["level"])
            if source in ["n_chars", "n_words"]:
                add(f"{source}_relative{window}", df[source] / mean.clip(lower=1))
    history = df.groupby("author", sort=False)["log_reactions"]
    for name, fn in [("mean", "mean"), ("std", "std"), ("min", "min"), ("max", "max")]:
        add("history_" + name, history.transform(
            lambda s: getattr(s.shift(1).expanding(min_periods=2), fn)()))
    add("level_to_followers", df["level"] - df["log_followers"])
    df["content_type"] = df["content_type"].fillna("unknown").astype(str)
    df = pd.concat([df, pd.DataFrame(added, index=df.index)], axis=1)
    extra = df.get("extra_text", pd.Series("", index=df.index)).fillna("")
    df["clean_text"] = (text + " " + extra).map(clean_text)
    profile = df.get("profile_text", pd.Series("EMPTY_PROFILE", index=df.index)).fillna("EMPTY_PROFILE")
    df["profile_text"] = profile.map(clean_text)
    numeric = BASE_NUMERIC + list(added) + PROFILE_NUM + POST_NUM
    df[numeric] = df[numeric].replace([np.inf, -np.inf], np.nan)
    return df, numeric
