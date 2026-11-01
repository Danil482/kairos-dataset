"""The label: how a post did against its own author's usual level.

    level  = median of log(1 + reactions) over the author's own EARLIER posts, at least 5
    resid  = log(1 + reactions of the post) - level
    class  = below (0) if resid < low, above (2) if resid > high, typical (1) otherwise

`low` and `high` are the 20th and 80th percentiles of the residuals of the train
posts: x0.577 and x3.182 of the author's usual level. The level reads earlier
posts only, so a post never enters the median it is judged against, and the
level of any post can be computed from the posts that precede it.

`python -m kairos.label` recomputes every label from the posts and checks it
against labels.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_HISTORY = 5
QUANTILES = (0.20, 0.80)
BELOW, TYPICAL, ABOVE = 0, 1, 2
CLASS_NAMES = {BELOW: "below", TYPICAL: "typical", ABOVE: "above"}
OFFICIAL_LABELS_SHA256 = "7a8ffd2c2b954569225469b69506fa38d61242cdb4202262453d8b4d84450501"


def author_level(posts: pd.DataFrame, min_history: int = MIN_HISTORY) -> pd.DataFrame:
    """Add `log_reactions`, `level`, `resid` and `hist_n` to an author's own posts.

    Takes the frame of `data.load_posts()` and returns it sorted by author and
    time. A post with fewer than `min_history` earlier posts has no level and no
    residual: the first five posts of every author are history, not labels.
    """
    out = posts.sort_values(["author", "posted_at"], kind="mergesort").copy()
    out["log_reactions"] = np.log1p(out["reactions"].clip(lower=0).astype(float))
    by_author = out.groupby("author")["log_reactions"]
    # shift(1) closes the window before the current post, so a post never enters
    # its own level.
    out["level"] = by_author.transform(
        lambda s: s.shift(1).expanding(min_periods=min_history).median())
    out["hist_n"] = by_author.cumcount()
    out["resid"] = out["log_reactions"] - out["level"]
    return out


def fit_cutoffs(train_resid: pd.Series) -> tuple[float, float]:
    """The 20th and 80th percentiles of the train residuals.

    split.json carries them rounded to six decimals. Both fall exactly on a
    residual that many posts share, and the rounded `low` moves 127 of those
    posts from typical to below. Refitting gives the exact values.
    """
    low, high = np.quantile(train_resid.dropna(), QUANTILES)
    return float(low), float(high)


def classify(resid: pd.Series, low: float, high: float) -> pd.Series:
    return pd.Series(np.where(resid < low, BELOW, np.where(resid > high, ABOVE, TYPICAL)),
                     index=resid.index, dtype="int8")


def relabel(posts: pd.DataFrame, train_authors) -> pd.DataFrame:
    """Every labelled post with its recomputed `label`, as in labels.csv."""
    levelled = author_level(posts)
    labelled = levelled[levelled["resid"].notna()].copy()
    is_train = labelled["author"].isin(set(train_authors))
    low, high = fit_cutoffs(labelled.loc[is_train, "resid"])
    labelled["label"] = classify(labelled["resid"], low, high)
    labelled.attrs["cutoffs"] = (low, high)
    return labelled


def verify_labels(released: pd.DataFrame, recomputed: pd.DataFrame, *,
                  labels_sha256: str) -> dict:
    """Verify keys and labels, permitting only pinned-release boundary roundoff.

    NumPy log1p differs by a few ULPs across platforms. The official release even
    assigns two classes to mathematically equal boundary ratios, so labels must
    remain unchanged. An exception requires its exact CSV checksum, adjacent
    classes and a residual within eight ULPs of the relevant cutoff. max_error
    reports the residual's distance to that cutoff, not an error in the label.
    """
    keys = ["author", "post"]
    for name, frame in (("released", released), ("recomputed", recomputed)):
        if not set(keys + ["label"]) <= set(frame):
            raise ValueError(f"{name} labels require author, post and label columns")
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError(f"{name} labels contain missing or duplicate post keys")
        if not frame.label.isin(CLASS_NAMES).all():
            raise ValueError(f"{name} labels must be 0, 1 or 2")
    both = released[keys + ["label"]].merge(
        recomputed, on=keys, how="outer", validate="one_to_one",
        suffixes=("_released", "_recomputed"), indicator=True)
    if not both["_merge"].eq("both").all():
        raise ValueError("released and recomputed eligible post key coverage differs")
    matches = both.label_released.eq(both.label_recomputed)
    report = {"exact_matches": int(matches.sum()), "boundary_roundoff_count": 0,
              "max_error": 0.0,
              "tolerance_policy": "8*spacing(max(abs(log_reactions),abs(level),1)); adjacent class at matching cutoff only",
              "labels_sha256": labels_sha256}
    if matches.all():
        return report
    different = both.loc[~matches]
    required = {"log_reactions", "level", "resid"}
    if not required <= set(different):
        raise ValueError("recomputed residuals and levels are required to check boundary roundoff")
    values = different[["log_reactions", "level", "resid"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("recomputed residuals and levels must be finite")
    cutoffs = recomputed.attrs.get("cutoffs")
    if cutoffs is None or len(cutoffs) != 2 or not np.isfinite(cutoffs).all() or cutoffs[0] >= cutoffs[1]:
        raise ValueError("recomputed labels require finite ordered cutoffs")
    low, high = cutoffs
    left = different.label_released.to_numpy(dtype=int)
    right = different.label_recomputed.to_numpy(dtype=int)
    at_low = ((left == BELOW) & (right == TYPICAL)) | ((left == TYPICAL) & (right == BELOW))
    at_high = ((left == TYPICAL) & (right == ABOVE)) | ((left == ABOVE) & (right == TYPICAL))
    tolerance = 8 * np.spacing(np.maximum(np.abs(values[:, :2]).max(axis=1), 1.0))
    error = np.where(at_low, np.abs(values[:, 2] - low), np.abs(values[:, 2] - high))
    if not ((at_low | at_high) & (error <= tolerance)).all():
        raise ValueError("released labels differ from recomputed labels outside numerical cutoff tolerance")
    if labels_sha256 != OFFICIAL_LABELS_SHA256:
        raise ValueError("boundary roundoff exception requires the unchanged official labels.csv SHA256")
    report["boundary_roundoff_count"] = int(len(different))
    report["max_error"] = float(error.max())
    return report


def main() -> None:
    import argparse
    import hashlib
    import io
    from pathlib import Path

    from kairos.data import DATA_DIR, load_posts, load_split

    ap = argparse.ArgumentParser(description="Check labels.csv against the posts.")
    ap.add_argument("--data", default=DATA_DIR, help="unpacked release (default: data/)")
    args = ap.parse_args()

    split = load_split(args.data)
    labels_blob = (Path(args.data) / "labels.csv").read_bytes()
    released = pd.read_csv(io.BytesIO(labels_blob))
    ours = relabel(load_posts(args.data), split["train"])
    low, high = ours.attrs["cutoffs"]
    shipped = split["cutoffs_on_log_residual"]
    print(f"cutoffs  low {low:+.6f} (split.json {shipped['low']:+.6f})  "
          f"high {high:+.6f} (split.json {shipped['high']:+.6f})")

    try:
        checked = verify_labels(released, ours, labels_sha256=hashlib.sha256(labels_blob).hexdigest())
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"labels.csv {len(released):,}  exact matches {checked['exact_matches']:,}  "
          f"boundary roundoff {checked['boundary_roundoff_count']:,}")
    if checked["boundary_roundoff_count"]:
        print(f"Platform portability differences accepted only at class boundaries "
              f"(max distance {checked['max_error']:.17g}); official labels unchanged")
    else:
        print("All labels match exactly; official labels unchanged")


if __name__ == "__main__":
    main()
