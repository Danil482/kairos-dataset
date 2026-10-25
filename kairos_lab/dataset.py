"""Release audit and past-only features. Training never receives test targets."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from kairos import features as baseline_features
from kairos.data import load_followers, load_labels, load_posts, load_split
from kairos.label import MIN_HISTORY, author_level, classify, fit_cutoffs, relabel
from .common import atomic_json, digest, identity, mark, stop_check, verified_marker

KEYS = ["author", "post"]
FORBIDDEN = {"label", "class", "split", "reactions", "log_reactions", "resid", "author", "post",
             "fold", "posted_at", "crawled_at", "text", "profile_text", "document_text"}
BASE_NUM = baseline_features.CONTENT_NUM + baseline_features.HISTORY_NUM + baseline_features.ARTEFACT_NUM


def canonical_text(value: str) -> str:
    return re.sub(r"⟨(PERSON|ORG|AUTHOR)_\d+⟩", r"⟨\1⟩", value)


def scalar_text(value) -> str:
    if isinstance(value, dict):
        return " ".join(scalar_text(v) for k, v in value.items()
                        if not any(x in k.lower() for x in ("url", "image", "photo", "urn", "avatar")))
    if isinstance(value, list):
        return " ".join(map(scalar_text, value))
    return str(value) if isinstance(value, (str, int, float)) else ""


def release_identity(data: Path) -> str:
    paths = [data / "labels.csv", data / "split.json"]
    paths += sorted((data / "posts").glob("AUTHOR_*.json"))
    paths += sorted((data / "authors").glob("AUTHOR_*.json"))
    manifest = data / "media/manifest.jsonl"
    if manifest.exists():
        paths.append(manifest)
    return identity({str(p.relative_to(data)): digest(p) for p in paths})


def extra_content(data: Path) -> tuple[pd.DataFrame, dict]:
    """Explicit pre-engagement allowlist; no current outcome fields are copied."""
    profiles, rows = {}, []
    schema = Counter()
    for path in sorted((data / "authors").glob("AUTHOR_*.json")):
        p = json.loads(path.read_text())
        allowed = [p[k] for k in ("about", "summary", "headline", "skills", "experience",
                                  "experiences", "education", "positions") if k in p]
        profiles[p.get("author", path.stem)] = canonical_text(scalar_text(allowed))[:12000]
    for path in sorted((data / "posts").glob("AUTHOR_*.json")):
        feed = json.loads(path.read_text())
        owner = feed["author"]
        for p in feed["posts"]:
            schema.update(p.keys())
            if not isinstance(p.get("author"), dict) or p["author"].get("code") != owner:
                continue
            document = p.get("document") or {}
            transcript = document.get("transcript", "") if isinstance(document, dict) else ""
            images = p.get("images") or []
            rows.append({"author": owner, "post": p["urn"],
                         "profile_text": profiles.get(owner, ""),
                         "document_text": canonical_text(scalar_text(transcript))[:30000],
                         "n_images": len(images) if isinstance(images, list) else int(bool(images)),
                         "has_video": int(bool(p.get("video"))),
                         "has_document": int(bool(document)),
                         "has_reshared": int(bool(p.get("resharedPost") or p.get("reshared_post")))})
    return pd.DataFrame(rows), {"post_keys": dict(schema), "profile_nonempty": sum(bool(x) for x in profiles.values())}


def history_features(frame: pd.DataFrame, work: Path | None = None) -> pd.DataFrame:
    """Strictly earlier timestamps, including tied-date exclusion, in author order."""
    df = frame.sort_values(["author", "posted_at", "post"], kind="stable").reset_index(drop=True).copy()
    text = df["text"].fillna("")
    df["n_exclamations"] = text.str.count("!")
    df["n_digits"] = text.str.count(r"\d")
    df["n_emoji"] = text.str.count(r"[\U0001F300-\U0001FAFF]")
    df["n_orgs"] = text.str.count(r"⟨ORG_\d+⟩")
    df["has_cta"] = text.str.lower().str.contains(r"\b(?:comment|share|join|register|agree|thoughts)\b", regex=True).astype(int)
    df["first_line_len"] = text.str.split("\n").str[0].str.len()
    df["cyrillic_fraction"] = text.str.count(r"[а-яА-ЯёЁ]") / text.str.len().clip(lower=1)
    df["upper_fraction"] = text.str.count(r"[A-ZА-Я]") / text.str.len().clip(lower=1)
    df["posted_year"] = df["posted_at"].dt.year
    df["posted_month"] = df["posted_at"].dt.month
    result = []
    for _, group in df.groupby("author", sort=False):
        if work is not None:
            stop_check(work)
        times = group["posted_at"].astype("int64").to_numpy() / 86400e9
        reactions = np.log1p(group["reactions"].clip(lower=0).to_numpy(float))
        lengths = group["n_chars"].to_numpy(float)
        types = group["content_type"].to_numpy()
        rows = []
        for i in range(len(group)):
            end = int(np.searchsorted(times, times[i], side="left"))
            past = reactions[:end]
            row = {"history_count": end,
                   "history_days": times[i] - times[0] if end else 0.0,
                   "previous_gap": times[i] - times[end - 1] if end else np.nan}
            if end:
                row.update(hist_mean=float(past.mean()), hist_median=float(np.median(past)),
                           hist_std=float(past.std()), hist_min=float(past.min()), hist_max=float(past.max()),
                           hist_q10=float(np.quantile(past, .1)), hist_q90=float(np.quantile(past, .9)),
                           length_vs_past=float(lengths[i] - np.median(lengths[:end])),
                           format_seen=float(np.mean(types[:end] == types[i])),
                           median_gap=float(np.median(np.diff(times[:end]))) if end > 1 else np.nan)
                same = past[types[:end] == types[i]]
                row["same_format_level"] = float((same.sum() + 5 * np.median(past)) / (len(same) + 5))
                for window in (5, 10, 20):
                    recent = past[-window:]
                    row[f"recent_{window}_mean"] = float(recent.mean())
                    row[f"recent_{window}_median"] = float(np.median(recent))
                    row[f"recent_{window}_std"] = float(recent.std())
                row["recent_trend"] = row["recent_5_mean"] - row["hist_mean"]
                row["length_ratio"] = float((lengths[i] + 1) / (np.median(lengths[:end]) + 1))
                for days in (7, 30, 90):
                    row[f"posts_last_{days}d"] = end - int(np.searchsorted(times[:end], times[i] - days))
            rows.append(row)
        result.extend(rows)
    hist = pd.DataFrame(result, index=df.index)
    # Baseline has collection-time missing-value imputation; extended models use NaN directly.
    df["age_days"] = (df["crawled_at"] - df["posted_at"]).dt.total_seconds() / 86400
    df["level"] = hist["hist_median"]
    df["hist_n"] = hist["history_count"]
    df["days_since_prev"] = hist["previous_gap"]
    # Strict prior-history spread replaces the original same-timestamp window.
    df = df.drop(columns=["hist_iqr"], errors="ignore")
    return pd.concat([df, hist], axis=1)


def assign_folds(train: pd.DataFrame, folds: int, seed: int) -> np.ndarray:
    out = np.full(len(train), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for k, (fit, valid) in enumerate(splitter.split(train, train["label"], groups=train["author"])):
        assert not set(train.iloc[fit].author) & set(train.iloc[valid].author)
        if set(train.iloc[fit].label) != {0, 1, 2}:
            raise ValueError("Each training fold must contain all three classes")
        out[valid] = k
    assert (out >= 0).all()
    return out


def audit_release_labels(posts: pd.DataFrame, labels: pd.DataFrame, train_authors) -> dict:
    """Require exact keys/classes, retrying only the audit's logarithm precision.

    Linux float64 log1p can move a residual a few ulps across a repeated cutoff.
    Extended log1p rounded once to float64 reproduces the release in that case.
    Targets and model features still use the unchanged release/original code.
    """
    def matches(frame):
        check = labels.merge(frame[KEYS + ["label"]], on=KEYS, how="outer",
                             suffixes=("", "_new"), indicator=True, validate="one_to_one")
        return (len(check) == len(labels) and check["_merge"].eq("both").all()
                and check.label.eq(check.label_new).all())

    recomputed = relabel(posts, train_authors)
    if matches(recomputed):
        return {"exact": True, "log1p_precision": "float64"}
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        raise ValueError("Release labels fail exact reproduction; extended precision unavailable")
    precise = posts.sort_values(["author", "posted_at"], kind="mergesort").copy()
    precise["log_reactions"] = np.log1p(
        precise.reactions.clip(lower=0).to_numpy(dtype=np.longdouble)).astype(np.float64)
    precise["level"] = precise.groupby("author").log_reactions.transform(
        lambda s: s.shift(1).expanding(min_periods=MIN_HISTORY).median())
    precise["resid"] = precise.log_reactions - precise.level
    precise = precise[precise.resid.notna()].copy()
    low, high = fit_cutoffs(precise.loc[precise.author.isin(set(train_authors)), "resid"])
    precise["label"] = classify(precise.resid, low, high)
    if not matches(precise):
        raise ValueError("Release labels fail exact reproduction (float64 and extended log1p)")
    return {"exact": True, "log1p_precision": "longdouble_rounded_to_float64",
            "cutoffs": [low, high]}


def prepare(data: Path, work: Path, config: dict) -> None:
    fingerprint = identity({"release": release_identity(data), "config": config})
    cache = work / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    marker = cache / "prepared.json"
    if verified_marker(marker, fingerprint):
        return
    stop_check(work)
    labels, split = load_labels(data), load_split(data)
    if labels.duplicated(KEYS).any() or not labels.label.isin([0, 1, 2]).all():
        raise ValueError("Invalid or duplicated release labels")
    if set(labels.split) != {"train", "test"}:
        raise ValueError("Expected official train/test split")
    for partition in ("train", "test"):
        selected = labels[labels.split == partition]
        if len(selected) != config["expected"][partition]:
            raise ValueError(f"Unexpected {partition} size: {len(selected)}")
        if set(selected.author) - set(split[partition]):
            raise ValueError("labels.csv disagrees with split.json")
    if set(split["train"]) & set(split["test"]):
        raise ValueError("Official author split overlaps")
    posts = load_posts(data)
    if posts.duplicated(KEYS).any() or posts.posted_at.isna().any():
        raise ValueError("Duplicated post keys or missing publication timestamps")
    label_audit = audit_release_labels(posts, labels, split["train"])
    atomic_json(cache / "label-audit.json", label_audit)
    base = baseline_features.build(author_level(posts), load_followers(data))
    baseline = labels[KEYS + ["split"]].merge(base[KEYS + BASE_NUM + ["content_type"]], on=KEYS, validate="one_to_one")
    baseline.to_parquet(cache / "baseline.parquet", index=False)
    extra, schema = extra_content(data)
    full = history_features(base.merge(extra, on=KEYS, validate="one_to_one"), work)
    full["text"] = full["text"].map(canonical_text)
    full[KEYS + ["posted_at", "reactions"]].to_parquet(cache / "history_outcomes.parquet", index=False)
    # Remove current outcomes before saving any modeling feature table.
    full = full.drop(columns=["reactions", "log_reactions", "resid"], errors="ignore")
    full.to_parquet(cache / "all_posts.parquet", index=False)
    joined = labels[KEYS + ["split"]].merge(full, on=KEYS, validate="one_to_one")
    train = joined[joined.split == "train"].drop(columns="split").reset_index(drop=True)
    train = train.merge(labels[labels.split == "train"][KEYS + ["label"]], on=KEYS, validate="one_to_one")
    train["fold"] = assign_folds(train, config["folds"], config["seed"])
    test = joined[joined.split == "test"].drop(columns="split").reset_index(drop=True)
    train.to_parquet(cache / "train.parquet", index=False)
    test.to_parquet(cache / "test.parquet", index=False)
    numeric = [c for c in full.select_dtypes(include="number").columns if c not in FORBIDDEN]
    atomic_json(cache / "feature_columns.json", {"numeric": numeric, "categorical": ["content_type"]})
    atomic_json(cache / "audit.json", {"release_fingerprint": fingerprint,
                "rows": {"train": len(train), "test": len(test), "own_posts": len(posts)},
                "labels_reproduced": True, "label_audit": label_audit, "test_targets_in_training": False,
                "tied_author_timestamps": int(posts.duplicated(["author", "posted_at"]).sum()),
                "schema": schema, "snapshot_features": ["log_followers", "age_days", "profile_text"]})
    paths = [cache / n for n in ["baseline.parquet", "all_posts.parquet", "history_outcomes.parquet", "train.parquet", "test.parquet",
                                "feature_columns.json", "audit.json", "label-audit.json"]]
    mark(marker, fingerprint, paths)
