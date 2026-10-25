"""Classical, non-deep features for the Kairos post-performance task.

Feature groups
--------------
content:  text/formatting, content type, calendar and media metadata
history:  author reaction history, posting cadence, and author-relative style
profile:  simple structured author-profile summaries (when supplied)
semantic: similarity to earlier posts and skill relevance (see semantic.py)
artefact: collection age and position in an author's history

Every feature derived from historical outcomes uses strictly earlier posts:
rolling and expanding statistics are shifted by one post first, so no
current-post reaction count or residual leaks into its own features.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
from emoji_extractor import Extractor

from kairos.label import MIN_HISTORY

# --------------------------------------------------------------------------- #
# Regexes
# --------------------------------------------------------------------------- #
# The release replaces links, handles and names with placeholders such as ⟨URL⟩,
# ⟨HANDLE⟩ and ⟨PERSON_3⟩, so the counters match placeholders as well as raw text.
_URL = re.compile(r"https?://|⟨URL⟩", flags=re.IGNORECASE)
_MENTION = re.compile(r"@\w+|⟨HANDLE⟩|⟨(?:PERSON|AUTHOR)[^⟩]*⟩", flags=re.IGNORECASE)
_HASHTAG = re.compile(r"#(?:\w+|⟨[^⟩]*⟩)")
_WORD = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", flags=re.UNICODE)
_BULLET_LINE = re.compile(r"^\s*(?:[-*•▪◦‣]|\d+[.)])\s+", flags=re.MULTILINE)
_REPEATED_PUNCT = re.compile(r"[!?.,;:…！？。，；：]{2,}")
_ELLIPSIS = re.compile(r"(?:\.{3,}|…+)")

# --------------------------------------------------------------------------- #
# Column groups (the baseline selects features by these names)
# --------------------------------------------------------------------------- #
MEDIA_NUM = [
    "n_media_items",
    "has_media",
    "has_image",
    "has_video",
    "has_document",
    "has_article",
]

# Style features compare the current post with the author's earlier median.
# "count" columns are compared on a log1p scale, "ratio" columns as they are.
_STYLE_COLUMNS = {
    "n_words": "count",
    "n_hashtags": "count",
    "n_emojis": "count",
    "punctuation_ratio": "ratio",
    "n_urls": "count",
    "uppercase_char_ratio": "ratio",
}

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
    "n_punctuation",
    "punctuation_ratio",
    "n_exclamations",
    "n_question_marks",
    "n_periods",
    "n_ellipses",
    "n_repeated_punctuation",
    "uppercase_char_ratio",
    "uppercase_word_ratio",
    "n_digits",
    "n_emojis",
    "n_unique_emojis",
    "has_emoji",
    "emojis_per_100_words",
    "n_paragraphs",
    "n_bullet_lines",
    "average_word_length",
    "n_unique_words",
    "unique_word_ratio",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_weekend",
    "month",
    "has_media_type",
    "has_image_type",
    "has_video_type",
    "has_document_type",
    "has_article_type",
    *MEDIA_NUM,
]
CONTENT_CAT = ["content_type"]

HISTORY_NUM = [
    "level",
    "hist_iqr",
    "days_since_prev",
    "log_followers",
    "hist_resid_mean_3",
    "hist_resid_mean_5",
    "hist_resid_mean_10",
    "hist_resid_mean_20",
    "hist_resid_std_5",
    "hist_resid_std_10",
    "hist_resid_std_20",
    "hist_log_reactions_mean_3",
    "hist_log_reactions_mean_5",
    "hist_log_reactions_mean_10",
    "hist_log_reactions_mean_20",
    "hist_positive_resid_share_5",
    "hist_positive_resid_share_10",
    "hist_positive_resid_share_20",
    "hist_negative_resid_share_10",
    "hist_resid_ewm_5",
    "hist_resid_momentum_3_10",
    "hist_resid_momentum_5_20",
    "hist_gap_median_5",
    "hist_gap_median_10",
    "hist_gap_std_10",
    "gap_vs_usual_log",
    "gap_ratio_to_usual",
    "hour_circular_deviation",
    *[
        name
        for column in _STYLE_COLUMNS
        for name in (f"style_delta_{column}", f"style_absdev_{column}")
    ],
]

PROFILE_NUM = [
    "profile_n_skills",
    "profile_has_skills",
    "profile_headline_chars",
    "profile_about_chars",
    "profile_n_positions",
]

WINDOWS = (5, 20)  # must match kairos.semantic.WINDOWS
KNN_KS = (3, 10)  # must match kairos.semantic.KNN_KS

SEMANTIC_NUM = [
    # history similarity
    "sim_prev_1",
    *[f"sim_{stat}_{w}" for w in WINDOWS for stat in ("mean", "max")],
    f"sim_centroid_{WINDOWS[-1]}",
    "sim_centroid_all",
    # skill relevance
    "skill_bm25",
    "skill_bm25_per_term",
    "skill_hit_share",
    "skill_emb_max",
    "skill_emb_centroid",
    # kNN over past posts (reactions of similar earlier posts)
    *[
        f"knn_{sim}_{stat}_reaction_{k}"
        for sim in ("emb", "bm25")
        for stat in ("mean", "max", "min")
        for k in KNN_KS
    ],
]

ARTEFACT_NUM = ["age_days", "hist_n"]
NUMERIC_FEATURES = CONTENT_NUM + HISTORY_NUM + PROFILE_NUM + SEMANTIC_NUM + ARTEFACT_NUM

# Content-type keywords that hint at attached media, used when no manifest is given.
_TYPE_PATTERNS = {
    "image": r"image|photo|picture",
    "video": r"video",
    "document": r"document|pdf|file|presentation",
    "article": r"article",
}
# Manifest flag -> content-type proxy used to fill it when no manifest is given.
_MEDIA_FLAG_PROXY = {
    "has_media": "has_media_type",
    "has_image": "has_image_type",
    "has_video": "has_video_type",
    "has_document": "has_document_type",
    "has_article": "has_article_type",
}

_REQUIRED_COLUMNS = {"author", "post", "posted_at", "text"}


# --------------------------------------------------------------------------- #
# Per-string text helpers
# --------------------------------------------------------------------------- #
def _punctuation_count(text: str) -> int:
    """Count Unicode punctuation characters (not emoji or other symbols)."""
    return sum(unicodedata.category(char).startswith("P") for char in text)


def _uppercase_char_ratio(text: str) -> float:
    """Share of alphabetic characters that are uppercase (0 for no letters)."""
    letters = [char for char in text if char.isalpha()]
    return sum(char.isupper() for char in letters) / max(len(letters), 1)


def _uppercase_word_ratio(text: str) -> float:
    words = _WORD.findall(text)
    return sum(word.upper() == word for word in words) / max(len(words), 1)


def _average_word_length(words: list[str]) -> float:
    return sum(map(len, words)) / len(words) if words else 0.0


def _paragraph_count(text: str) -> int:
    return sum(bool(part.strip()) for part in re.split(r"\n\s*\n", text.strip()))


# --------------------------------------------------------------------------- #
# Strictly-prior (leak-free) history helpers
# --------------------------------------------------------------------------- #
def _prior_rolling(
    df: pd.DataFrame,
    values: pd.Series,
    window: int,
    stat: str,
    min_periods: int = 1,
) -> pd.Series:
    """Rolling mean/median/std over each author's previous posts (missing -> 0)."""
    author = df["author"]
    lagged = values.groupby(author, sort=False).shift(1)

    def rolling_stat(series: pd.Series) -> pd.Series:
        rolling = series.rolling(window=window, min_periods=min_periods)
        return rolling.std(ddof=0) if stat == "std" else getattr(rolling, stat)()

    return lagged.groupby(author, sort=False).transform(rolling_stat).fillna(0.0)


def _prior_expanding(
    df: pd.DataFrame, column: str, stat, min_periods: int
) -> pd.Series:
    """``stat`` of an expanding window over each author's previous posts.

    ``stat`` is a function of the expanding window, e.g. ``lambda e: e.median()``.
    """
    return df.groupby("author", sort=False)[column].transform(
        lambda values: stat(values.shift(1).expanding(min_periods=min_periods))
    )


# --------------------------------------------------------------------------- #
# Feature blocks (each one adds columns to ``df`` in place)
# --------------------------------------------------------------------------- #
def _add_text_features(df: pd.DataFrame) -> None:
    text = df["text"]

    # Basic counts.
    df["n_chars"] = text.str.len().astype("int32")
    df["n_words"] = text.str.split().str.len().fillna(0).astype("int32")
    df["n_lines"] = text.str.count("\n") + 1
    df["n_hashtags"] = text.str.count(_HASHTAG).astype("int32")
    df["n_mentions"] = text.str.count(_MENTION).astype("int32")
    df["n_urls"] = text.str.count(_URL).astype("int32")
    df["has_question"] = text.str.contains("?", regex=False).astype("int8")
    df["is_textless"] = (df["n_chars"] == 0).astype("int8")

    # Punctuation and casing. Ratios of empty strings are 0, never NaN.
    df["n_punctuation"] = text.map(_punctuation_count).astype("int32")
    df["punctuation_ratio"] = df["n_punctuation"] / df["n_chars"].clip(lower=1)
    df["n_exclamations"] = text.str.count(r"[!！]").astype("int32")
    df["n_question_marks"] = text.str.count(r"[?？]").astype("int32")
    df["n_periods"] = text.str.count(r"[.．]").astype("int32")
    df["n_ellipses"] = text.map(lambda t: len(_ELLIPSIS.findall(t))).astype("int32")
    df["n_repeated_punctuation"] = text.map(
        lambda t: len(_REPEATED_PUNCT.findall(t))
    ).astype("int32")
    df["uppercase_char_ratio"] = text.map(_uppercase_char_ratio)
    df["uppercase_word_ratio"] = text.map(_uppercase_word_ratio)
    df["n_digits"] = text.map(lambda t: sum(char.isdigit() for char in t)).astype(
        "int32"
    )

    # Vocabulary.
    words = text.str.casefold().map(_WORD.findall)
    df["n_unique_words"] = words.map(lambda w: len(set(w))).astype("int32")
    df["average_word_length"] = words.map(_average_word_length)
    df["unique_word_ratio"] = df["n_unique_words"] / words.map(len).clip(lower=1)

    # Emoji.
    ext = Extractor()
    emoji_counts = text.map(ext.count_emoji)

    df["n_emojis"] = emoji_counts.map(lambda c: sum(c.values())).astype("int32")
    df["n_unique_emojis"] = emoji_counts.map(len).astype("int32")
    df["has_emoji"] = (df["n_emojis"] > 0).astype("int8")
    df["emojis_per_100_words"] = 100.0 * df["n_emojis"] / df["n_words"].clip(lower=1)

    # Layout.
    df["n_paragraphs"] = text.map(_paragraph_count).astype("int32")
    df["n_bullet_lines"] = text.map(lambda t: len(_BULLET_LINE.findall(t))).astype(
        "int32"
    )


def _add_calendar_features(df: pd.DataFrame) -> None:
    """Raw calendar values plus cyclic encodings of hour and weekday."""
    df["hour"] = df["posted_at"].dt.hour
    df["weekday"] = df["posted_at"].dt.weekday
    df["month"] = df["posted_at"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7.0)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7.0)
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype("int8")


def _add_content_type_flags(df: pd.DataFrame) -> None:
    """Media flags guessed from ``content_type``; the fallback when no manifest exists."""
    content_type = df.get("content_type", pd.Series("unknown", index=df.index))
    content_type = content_type.fillna("unknown").astype(str).str.casefold()
    for kind, pattern in _TYPE_PATTERNS.items():
        df[f"has_{kind}_type"] = content_type.str.contains(pattern, regex=True).astype(
            "int8"
        )
    type_columns = [f"has_{kind}_type" for kind in _TYPE_PATTERNS]
    df["has_media_type"] = df[type_columns].max(axis=1).astype("int8")


def _add_age_and_scale(df: pd.DataFrame, followers: pd.Series) -> None:
    """Collection age (days between posting and crawl) and author audience size."""
    if "crawled_at" in df.columns:
        age_days = (df["crawled_at"] - df["posted_at"]).dt.total_seconds() / 86400.0
        median = age_days.median()
        df["age_days"] = age_days.fillna(0.0 if pd.isna(median) else median)
    else:
        df["age_days"] = np.nan

    df["log_followers"] = np.log1p(df["author"].map(followers).fillna(0).clip(lower=0))


def _outcome_series(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return (log_reactions, residual), falling back to NaN when unavailable."""
    nothing = pd.Series(np.nan, index=df.index, dtype=float)

    if "log_reactions" in df.columns:
        log_reactions = pd.to_numeric(df["log_reactions"], errors="coerce")
    elif "reactions" in df.columns:
        reactions = pd.to_numeric(df["reactions"], errors="coerce")
        log_reactions = np.log1p(reactions.clip(lower=0))
    else:
        log_reactions = nothing

    residual = (
        pd.to_numeric(df["resid"], errors="coerce")
        if "resid" in df.columns
        else nothing
    )
    return log_reactions, residual


def _add_outcome_history(df: pd.DataFrame) -> None:
    """Rolling reaction-level and residual statistics over earlier posts."""
    log_reactions, residual = _outcome_series(df)

    for window in (3, 5, 10, 20):
        df[f"hist_resid_mean_{window}"] = _prior_rolling(df, residual, window, "mean")
        df[f"hist_log_reactions_mean_{window}"] = _prior_rolling(
            df, log_reactions, window, "mean"
        )
    for window in (5, 10, 20):
        df[f"hist_resid_std_{window}"] = _prior_rolling(
            df, residual, window, "std", min_periods=2
        )

    # Share of earlier posts that beat / missed the author's expected level.
    known = residual.notna()
    positive = (residual > 0).astype(float).where(known)
    negative = (residual < 0).astype(float).where(known)
    for window in (5, 10, 20):
        df[f"hist_positive_resid_share_{window}"] = _prior_rolling(
            df, positive, window, "mean"
        )
    df["hist_negative_resid_share_10"] = _prior_rolling(df, negative, 10, "mean")

    df["hist_resid_ewm_5"] = (
        residual.groupby(df["author"], sort=False)
        .transform(
            lambda values: values.shift(1)
            .ewm(span=5, adjust=False, min_periods=1)
            .mean()
        )
        .fillna(0.0)
    )
    df["hist_resid_momentum_3_10"] = df["hist_resid_mean_3"] - df["hist_resid_mean_10"]
    df["hist_resid_momentum_5_20"] = df["hist_resid_mean_5"] - df["hist_resid_mean_20"]


def _add_cadence_features(df: pd.DataFrame) -> None:
    """Gap since the previous post, versus the author's usual gap and usual hour."""
    previous_post = df.groupby("author", sort=False)["posted_at"].shift(1)
    gap = (df["posted_at"] - previous_post).dt.total_seconds() / 86400.0
    df["days_since_prev"] = gap.fillna(-1.0)  # -1 marks an author's first post

    valid_gaps = df["days_since_prev"].where(df["days_since_prev"] >= 0)
    df["hist_gap_median_5"] = _prior_rolling(df, valid_gaps, 5, "median")
    df["hist_gap_median_10"] = _prior_rolling(df, valid_gaps, 10, "median")
    df["hist_gap_std_10"] = _prior_rolling(df, valid_gaps, 10, "std", min_periods=2)

    gap_days = df["days_since_prev"].clip(lower=0)
    usual_gap = df["hist_gap_median_10"]
    df["gap_vs_usual_log"] = np.log1p(gap_days) - np.log1p(usual_gap)
    df["gap_ratio_to_usual"] = gap_days / (usual_gap + 1.0)

    # Posting hour vs the author's earlier hours: 1 - cosine similarity between
    # this post's hour on the clock circle and the mean earlier hour vector.
    mean_sin = _prior_expanding(df, "hour_sin", lambda e: e.mean(), min_periods=1)
    mean_cos = _prior_expanding(df, "hour_cos", lambda e: e.mean(), min_periods=1)
    norm = np.sqrt(mean_sin.pow(2) + mean_cos.pow(2))
    dot = df["hour_sin"] * mean_sin + df["hour_cos"] * mean_cos
    cosine = (dot / norm.clip(lower=1e-9)).clip(-1.0, 1.0)
    df["hour_circular_deviation"] = (1.0 - cosine).where(norm > 1e-6, 0.0).fillna(0.0)


def _add_style_features(df: pd.DataFrame) -> None:
    """Signed and absolute deviation from the author's earlier median style."""
    for column, kind in _STYLE_COLUMNS.items():
        prior_median = _prior_expanding(
            df, column, lambda e: e.median(), min_periods=MIN_HISTORY
        )
        if kind == "count":
            current = np.log1p(df[column].clip(lower=0))
            reference = np.log1p(prior_median.clip(lower=0))
        else:
            current, reference = df[column], prior_median
        delta = (current - reference).fillna(0.0)
        df[f"style_delta_{column}"] = delta
        df[f"style_absdev_{column}"] = delta.abs()


def _add_level_and_spread(df: pd.DataFrame) -> None:
    """Fill ``level``/``hist_n`` if absent and add the earlier-posts IQR."""
    if "level" not in df.columns:
        df["level"] = np.nan
    if "hist_n" not in df.columns:
        df["hist_n"] = df.groupby("author", sort=False).cumcount()

    if "log_reactions" in df.columns:

        def prior_quantile(q: float) -> pd.Series:
            return _prior_expanding(
                df, "log_reactions", lambda e: e.quantile(q), min_periods=MIN_HISTORY
            )

        df["hist_iqr"] = (prior_quantile(0.75) - prior_quantile(0.25)).fillna(0.0)
    elif "hist_iqr" not in df.columns:
        df["hist_iqr"] = 0.0


# --------------------------------------------------------------------------- #
# Optional profile / media tables
# --------------------------------------------------------------------------- #
def _merge_profile_features(df: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Join per-author profile columns onto every post (last duplicate wins)."""
    profiles = profiles.copy()
    if "author" in profiles.columns:
        profiles = profiles.set_index("author")
    profiles.index.name = "author"

    columns = [col for col in PROFILE_NUM if col in profiles.columns]
    if not columns:
        return df
    profiles = profiles.loc[~profiles.index.duplicated(keep="last"), columns]
    return df.drop(columns=columns, errors="ignore").join(profiles, on="author")


def _merge_post_features(
    df: pd.DataFrame, table: pd.DataFrame, wanted: list[str], name: str
):
    """Join per-post columns keyed by (author, post); last duplicate wins."""
    keys = ["author", "post"]
    if not set(keys) <= set(table.columns):
        raise ValueError(f"{name} must contain 'author' and 'post' columns")

    columns = [col for col in wanted if col in table.columns]
    if not columns:
        return df
    table = table[keys + columns].drop_duplicates(keys, keep="last")
    return df.drop(columns=columns, errors="ignore").merge(
        table, on=keys, how="left", sort=False, validate="many_to_one"
    )


def _add_optional_features(
    df: pd.DataFrame,
    profile_features: pd.DataFrame | None,
    media_features: pd.DataFrame | None,
    semantic_features: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge the optional tables and guarantee a stable column schema."""
    if profile_features is not None:
        df = _merge_profile_features(df, profile_features)
    if media_features is not None:
        df = _merge_post_features(df, media_features, MEDIA_NUM, "media_features")
    if semantic_features is not None:
        df = _merge_post_features(
            df, semantic_features, SEMANTIC_NUM, "semantic_features"
        )

    # Semantic columns stay NaN when unknown (no history, no skills); the model imputes.
    missing = [c for c in PROFILE_NUM + SEMANTIC_NUM + MEDIA_NUM if c not in df.columns]
    df = df.assign(**dict.fromkeys(missing, np.nan))

    if media_features is not None:
        # With a manifest, "no manifest row" means "no attached files".
        df[MEDIA_NUM] = df[MEDIA_NUM].fillna(0)
    else:
        # Without one, fall back to the content-type guess for the flags only;
        # the item count stays missing because it cannot be guessed.
        for flag, proxy in _MEDIA_FLAG_PROXY.items():
            df[flag] = df[flag].fillna(df[proxy])
    return df


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def build(
    levelled: pd.DataFrame,
    followers: pd.Series,
    profile_features: pd.DataFrame | None = None,
    media_features: pd.DataFrame | None = None,
    semantic_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build all non-deep features for every own post.

    Parameters
    ----------
    levelled:
        Output of ``label.author_level`` (``log_reactions``, ``level``, ``resid``,
        ``hist_n``). Sorted by author/time internally; the input row order is
        restored on return.
    followers:
        Author ID -> follower count, as returned by ``data.load_followers``.
    profile_features:
        Optional output of ``data.load_author_features`` (indexed by author, or
        with an ``author`` column).
    media_features:
        Optional output of ``data.load_media_features``, keyed by author and post.
    semantic_features:
        Optional output of ``semantic.build_semantic_features``, keyed by author and post.
    """
    missing = sorted(_REQUIRED_COLUMNS - set(levelled.columns))
    if missing:
        raise ValueError(f"levelled is missing required columns: {missing}")

    df = levelled.copy()
    df["__feature_order"] = np.arange(len(df))
    df = df.sort_values(["author", "posted_at"], kind="mergesort")
    df["text"] = df["text"].fillna("").astype(str)

    _add_text_features(df)
    _add_calendar_features(df)
    _add_content_type_flags(df)
    _add_age_and_scale(df, followers)
    _add_outcome_history(df)
    _add_cadence_features(df)
    _add_style_features(df)
    _add_level_and_spread(df)
    df = _add_optional_features(df, profile_features, media_features, semantic_features)

    return df.sort_values("__feature_order", kind="mergesort").drop(
        columns="__feature_order"
    )
