"""Tier 1 features: author-relative tabular signals on top of the baseline.

Everything that references author history is computed with shift(1) + expanding
so the current post never sees itself. The output table has one row per own
post and includes every column the baseline uses, plus the Tier 1 additions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kairos import features as base_features
from kairos.data import DATA_DIR, load_followers, load_posts
from kairos.label import MIN_HISTORY, author_level


CYCLIC = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
          "day_of_month_sin", "day_of_month_cos"]

AUTHOR_DELTAS = ["n_chars_delta", "n_words_delta", "n_hashtags_delta",
                 "n_mentions_delta", "n_urls_delta", "hour_delta_circular",
                 "weekday_match", "content_type_rarity",
                 "is_textless_vs_author", "has_question_vs_author"]

ROLLING = ["roll_mean_5", "roll_std_5", "roll_mean_20", "roll_trend_5",
           "roll_mean_all", "roll_std_all"]

ACTIVITY = ["posts_per_week_90d", "cadence_cv_30d", "days_since_first_post",
            "hist_n_log"]

ARTEFACT_EXTRA = ["is_recent_post", "age_days_log"]

TIER1_NUMERIC = (base_features.CONTENT_NUM + base_features.HISTORY_NUM
                 + base_features.ARTEFACT_NUM
                 + CYCLIC + AUTHOR_DELTAS + ROLLING + ACTIVITY + ARTEFACT_EXTRA)
TIER1_CAT = base_features.CONTENT_CAT


def _cyclic(series: pd.Series, period: int, name: str) -> pd.DataFrame:
    rad = 2 * np.pi * series.astype(float) / period
    return pd.DataFrame({f"{name}_sin": np.sin(rad), f"{name}_cos": np.cos(rad)},
                        index=series.index)


def _expanding_median(group: pd.Series, min_periods: int = 1) -> pd.Series:
    return group.shift(1).expanding(min_periods=min_periods).median()


def _expanding_mode(group: pd.Series, min_periods: int = 1) -> pd.Series:
    # mode over categorical history: fastest exact approach is per-author
    # cumulative counts; here we fall back to a vectorized "most frequent so far"
    # computed inside groupby. For small groups this is fine.
    def _mode(s: pd.Series) -> pd.Series:
        hist = s.shift(1)
        out = []
        counts: dict = {}
        for v in hist:
            if pd.isna(v):
                out.append(np.nan)
            else:
                counts[v] = counts.get(v, 0) + 1
                out.append(max(counts, key=counts.get))
        return pd.Series(out, index=s.index)

    return group.transform(_mode) if hasattr(group, "transform") else _mode(group)


def _rolling_trend(window: pd.Series) -> float:
    y = window.values.astype(float)
    if len(y) < 2 or np.all(np.isnan(y)):
        return np.nan
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan
    x = np.arange(len(y))[mask]
    yv = y[mask]
    # slope of simple linear fit
    xm = x.mean()
    ym = yv.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - xm) * (yv - ym)).sum() / denom)


def build(levelled: pd.DataFrame, followers: pd.Series) -> pd.DataFrame:
    """Levelled = author_level(load_posts(...)). Followers = load_followers(...)."""
    df = base_features.build(levelled, followers).copy()

    # ---------- Cyclic encodings
    df = pd.concat([df,
                    _cyclic(df["hour"], 24, "hour"),
                    _cyclic(df["weekday"], 7, "weekday"),
                    _cyclic(df["posted_at"].dt.day, 31, "day_of_month")], axis=1)

    # ---------- Author-relative deltas (expanding median of past posts)
    g = df.groupby("author", sort=False)
    for col in ["n_chars", "n_words", "n_hashtags", "n_mentions", "n_urls"]:
        past_median = g[col].transform(lambda s: _expanding_median(s, 1))
        df[f"{col}_delta"] = (df[col] - past_median).fillna(0.0)

    past_textless = g["is_textless"].transform(lambda s: s.shift(1).expanding(1).mean())
    df["is_textless_vs_author"] = (df["is_textless"] - past_textless).fillna(0.0)
    past_question = g["has_question"].transform(lambda s: s.shift(1).expanding(1).mean())
    df["has_question_vs_author"] = (df["has_question"] - past_question).fillna(0.0)

    # Circular hour delta against author's "typical hour" (vector mean of sin/cos)
    past_hour_sin = g["hour_sin"].transform(lambda s: s.shift(1).expanding(1).mean())
    past_hour_cos = g["hour_cos"].transform(lambda s: s.shift(1).expanding(1).mean())
    # typical hour direction
    df["hour_delta_circular"] = np.arctan2(
        df["hour_sin"] - past_hour_sin,
        df["hour_cos"] - past_hour_cos,
    ).fillna(0.0)

    # Weekday match: most frequent prior weekday for the author
    past_wd_mode = _expanding_mode(g["weekday"])
    df["weekday_match"] = (df["weekday"] == past_wd_mode).astype(int)
    df.loc[past_wd_mode.isna(), "weekday_match"] = 0

    # Content type rarity: share of this content_type in prior history
    def _cat_rarity(sub: pd.DataFrame) -> pd.Series:
        hist = sub["content_type"].shift(1)
        rarity = pd.Series(np.nan, index=sub.index)
        counts: dict = {}
        total = 0
        for idx, cur_ct, prev_ct in zip(sub.index, sub["content_type"], hist):
            if total > 0:
                share = counts.get(cur_ct, 0) / total
                rarity.loc[idx] = 1.0 - share  # higher = rarer for the author
            if not pd.isna(prev_ct):
                counts[prev_ct] = counts.get(prev_ct, 0) + 1
                total += 1
        return rarity.fillna(0.5)

    df["content_type_rarity"] = (df.groupby("author", sort=False, group_keys=False)
                                 .apply(_cat_rarity, include_groups=False))

    # ---------- Rolling reaction stats (based on PRIOR posts only)
    rr = g["log_reactions"]
    df["roll_mean_5"] = rr.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0.0)
    df["roll_std_5"] = rr.transform(lambda s: s.shift(1).rolling(5, min_periods=2).std()).fillna(0.0)
    df["roll_mean_20"] = rr.transform(lambda s: s.shift(1).rolling(20, min_periods=1).mean()).fillna(0.0)
    df["roll_mean_all"] = rr.transform(lambda s: s.shift(1).expanding(1).mean()).fillna(0.0)
    df["roll_std_all"] = rr.transform(lambda s: s.shift(1).expanding(2).std()).fillna(0.0)

    def _trend_rolling(s: pd.Series) -> pd.Series:
        return s.shift(1).rolling(5, min_periods=2).apply(_rolling_trend, raw=False)

    df["roll_trend_5"] = rr.transform(_trend_rolling).fillna(0.0)

    # ---------- Author activity
    df["days_since_first_post"] = (df["posted_at"] - g["posted_at"].transform("min")
                                   ).dt.total_seconds() / 86400

    # Posts per week in the previous 90 days
    def _ppw_90d(sub: pd.DataFrame) -> pd.Series:
        times = sub["posted_at"].values.astype("datetime64[ns]")
        out = np.zeros(len(sub))
        for i in range(len(sub)):
            if i == 0:
                continue
            cutoff = times[i] - np.timedelta64(90, "D")
            # count prior posts in window
            n = np.sum((times[:i] >= cutoff) & (times[:i] < times[i]))
            out[i] = n / (90 / 7)
        return pd.Series(out, index=sub.index)

    df["posts_per_week_90d"] = (df.groupby("author", sort=False, group_keys=False)
                                .apply(_ppw_90d, include_groups=False))

    # Coefficient of variation of inter-post intervals, last 30 days
    def _cadence_cv(sub: pd.DataFrame) -> pd.Series:
        times = sub["posted_at"].values.astype("datetime64[ns]")
        out = np.zeros(len(sub))
        for i in range(len(sub)):
            if i < 2:
                continue
            cutoff = times[i] - np.timedelta64(30, "D")
            prev = times[:i][times[:i] >= cutoff]
            if len(prev) < 2:
                continue
            diffs = np.diff(prev).astype("timedelta64[s]").astype(float)
            if diffs.mean() > 0:
                out[i] = diffs.std() / diffs.mean()
        return pd.Series(out, index=sub.index)

    df["cadence_cv_30d"] = (df.groupby("author", sort=False, group_keys=False)
                            .apply(_cadence_cv, include_groups=False))

    df["hist_n_log"] = np.log1p(df["hist_n"].fillna(0))

    # ---------- Artefact extras
    df["is_recent_post"] = (df["age_days"] < 30).astype(int)
    df["age_days_log"] = np.log1p(df["age_days"].clip(lower=0))

    return df


def materialise(data_dir: Path | str = DATA_DIR,
                cache_path: Path | str = "cache/tier1_features.parquet",
                force: bool = False) -> pd.DataFrame:
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    levelled = author_level(load_posts(data_dir))
    df = build(levelled, load_followers(data_dir))
    df.to_parquet(cache_path, index=False)
    return df


if __name__ == "__main__":
    df = materialise(force=True)
    print(f"wrote cache/tier1_features.parquet: {len(df):,} rows, "
          f"{df.shape[1]} columns")
    print("columns:", list(df.columns))
