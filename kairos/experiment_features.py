"""Causal tabular features and source-preserving text for the Lab 1 experiments.

The history includes the initial unlabelled posts. Only earlier own posts enter
historical features; current reactions are retained in history for later posts,
never in the labelled feature table. Test targets are used only for integrity.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from kairos import features
from kairos.label import CLASS_NAMES, MIN_HISTORY, relabel, verify_labels

KEYS = ["author", "post"]
CATEGORICAL = ["content_type"]
DOCUMENT_MAX_CHARS = 20_000
FEATURE_GROUPS = {
    "baseline": features.CONTENT_NUM + features.HISTORY_NUM + features.ARTEFACT_NUM,
    "recent5": ["recent5_delta"],
    "recent_extra": ["recent10_delta", "recent5_vs_previous5", "has_10_history"],
    "gap": ["gap_surprise"],
    "style": ["n_images", "shared_n_images", "n_document_pages", "shared_document_pages",
              "caps_fraction", "n_emoji_codepoints", "n_exclamations", "n_repeated_letter_runs",
              "shared_n_chars", "content_n_chars", "length_delta", "caps_delta",
              "urls_delta", "mentions_delta", "past_same_type_fraction"],
    "mentions": ["n_person_mentions", "n_company_mentions", "has_shared",
                 "shared_from_person", "shared_from_company", "person_new_count",
                 "person_seen_fraction", "org_new_count", "org_seen_fraction"],
}
NUMERIC = list(dict.fromkeys(name for group in FEATURE_GROUPS.values() for name in group))
TEXT_SOURCES = ["text", "shared_text", "article_title", "document_title", "document_text",
                "shared_article_title", "shared_document_title", "shared_document_text"]
_PSEUDONYM = re.compile(r"⟨(PERSON|ORG|AUTHOR)_[^⟩]+⟩")
_CODE = re.compile(r"⟨((?:PERSON|ORG|AUTHOR)_[^⟩]+)⟩")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
_REPEATED = re.compile(r"([^\W\d_])\1{2,}", re.IGNORECASE)


def normalize_text(value: str) -> str:
    """Remove pseudonym IDs while preserving genuine empty text."""
    return _PSEUDONYM.sub(r"⟨\1⟩", value).strip()


def _text(value) -> str:
    return value if isinstance(value, str) else ""


def _object(value) -> dict:
    return value if isinstance(value, dict) else {}


def _list(value) -> list:
    return value if isinstance(value, list) else []


def _document(document: dict) -> tuple[str, str, bool]:
    transcript = document.get("transcript")
    text = ("\n".join(x for x in transcript if isinstance(x, str))
            if isinstance(transcript, list) else _text(transcript))
    truncated = len(text) > DOCUMENT_MAX_CHARS
    return _text(document.get("title")), text[:DOCUMENT_MAX_CHARS], truncated


def _pages(document: dict) -> float:
    if not document:
        return 0.0
    try:
        value = float(document.get("totalPageCount"))
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) and value >= 0 else np.nan


def extract_post(owner: str, post: dict, crawled_at=None) -> dict:
    """Extract allowed sources; no nested engagement counters or profile text."""
    posted_at = pd.to_datetime(post.get("postedDateTimestamp"), unit="ms", utc=True,
                               errors="coerce")
    shared = _object(post.get("resharedPost"))
    shared_time = pd.to_datetime(shared.get("postedDateTimestamp"), unit="ms", utc=True,
                                 errors="coerce")
    future_shared = bool(shared and pd.notna(shared_time) and pd.notna(posted_at)
                         and shared_time > posted_at)
    shared_time_missing = bool(shared and pd.isna(shared_time))
    if future_shared:
        shared = {}
    document = _object(post.get("document"))
    shared_document = _object(shared.get("document"))
    title, document_text, truncated = _document(document)
    shared_title, shared_document_text, shared_truncated = _document(shared_document)
    raw_text = _text(post.get("text"))
    raw_sources = {
        "text": raw_text, "shared_text": _text(shared.get("text")),
        "article_title": _text(_object(post.get("article")).get("title")),
        "document_title": title, "document_text": document_text,
        "shared_article_title": _text(_object(shared.get("article")).get("title")),
        "shared_document_title": shared_title, "shared_document_text": shared_document_text,
    }
    sources = {name: normalize_text(value) for name, value in raw_sources.items()}
    content = "\n\n".join(f"[{name}]\n{value}" for name, value in sources.items() if value)
    people = _list(post.get("mentions")) + _list(shared.get("mentions"))
    companies = _list(post.get("companyMentions")) + _list(shared.get("companyMentions"))
    codes = set(_CODE.findall("\n".join(raw_sources.values())))
    codes.update(item["code"] for item in people + companies
                 if isinstance(item, dict) and isinstance(item.get("code"), str))
    publisher_org = _object(shared.get("company")).get("code")
    publisher_person = _object(shared.get("author")).get("code")
    if isinstance(publisher_org, str) and publisher_org.startswith("ORG_"):
        codes.add(publisher_org)
    if isinstance(publisher_person, str) and (publisher_person.startswith("PERSON_") or publisher_person == owner):
        codes.add(publisher_person)
    letters = [char for char in raw_text if char.isalpha()]
    return {
        "author": owner, "post": post["urn"], "posted_at": posted_at,
        "crawled_at": crawled_at, "reactions": int(post.get("totalReactionCount") or 0),
        **sources, "content_text": content,
        "content_type": post.get("contentType") or "unknown",
        # Baseline text counts retain the original unnormalised own-text definition.
        "n_chars": len(raw_text), "n_words": len(raw_text.split()),
        "n_lines": raw_text.count("\n") + 1,
        "n_hashtags": len(features._HASHTAG.findall(raw_text)),
        "n_mentions": len(features._MENTION.findall(raw_text)),
        "n_urls": len(features._URL.findall(raw_text)),
        "has_question": int("?" in raw_text), "is_textless": int(not raw_text),
        "n_images": len(_list(post.get("images"))),
        "shared_n_images": len(_list(shared.get("images"))),
        "n_document_pages": _pages(document), "shared_document_pages": _pages(shared_document),
        "caps_fraction": sum(char.isupper() for char in letters) / max(1, len(letters)),
        "n_emoji_codepoints": len(_EMOJI.findall(raw_text)),
        "n_exclamations": raw_text.count("!"),
        "n_repeated_letter_runs": len(_REPEATED.findall(raw_text)),
        "shared_n_chars": len(raw_sources["shared_text"]),
        "content_n_chars": sum(len(value) for value in sources.values()),
        "n_person_mentions": len(people), "n_company_mentions": len(companies),
        "has_shared": int(bool(shared)),
        "shared_from_person": int(bool(_object(shared.get("author")))),
        "shared_from_company": int(bool(_object(shared.get("company")))),
        "person_codes": tuple(sorted(code for code in codes if code.startswith(("PERSON_", "AUTHOR_")))),
        "org_codes": tuple(sorted(code for code in codes if code.startswith("ORG_"))),
        "future_shared_excluded": future_shared, "shared_timestamp_missing": shared_time_missing,
        "document_truncated": truncated, "shared_document_truncated": shared_truncated,
    }


def _validate_keys(frame: pd.DataFrame, name: str) -> None:
    if not set(KEYS) <= set(frame):
        raise ValueError(f"{name} requires author and post columns")
    if frame[KEYS].isna().any().any() or frame[KEYS].eq("").any().any():
        raise ValueError(f"{name} has a missing author or post key")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{name} has duplicate (author, post) keys")


def _ordered_history(history: pd.DataFrame) -> pd.DataFrame:
    _validate_keys(history, "history")
    result = history.sort_values(["author", "posted_at"], kind="mergesort").reset_index(drop=True).copy()
    if result.posted_at.isna().any():
        raise ValueError("own posts have missing publication timestamps; strict history is undefined")
    if result.duplicated(["author", "posted_at"]).any():
        raise ValueError("own posts have tied timestamps within an author; strict history is undefined")
    if "reactions" in result:
        result["log_reactions"] = np.log1p(result.reactions.clip(lower=0).astype(float))
    if not np.isfinite(result.log_reactions).all():
        raise ValueError("history log_reactions must be finite")
    past = result.groupby("author", sort=False).log_reactions
    result["level"] = past.transform(lambda s: s.shift(1).expanding(min_periods=MIN_HISTORY).median())
    result["hist_n"] = result.groupby("author", sort=False).cumcount()
    return result


def build_features(history: pd.DataFrame, followers: pd.Series | None = None) -> pd.DataFrame:
    """Build an allowlisted table without inspecting labels or fitting transforms.

    Input rows come from extract_post, with datetime timestamps and either reactions
    or log_reactions. Previously computed levels are always recomputed causally.
    """
    df = _ordered_history(history)
    past = df.groupby("author", sort=False).log_reactions
    recent5 = past.transform(lambda s: s.shift(1).rolling(5, min_periods=5).median())
    recent10 = past.transform(lambda s: s.shift(1).rolling(10, min_periods=10).median())
    previous5 = past.transform(lambda s: s.shift(6).rolling(5, min_periods=5).median())
    df["recent5_delta"] = recent5 - df.level
    df["recent10_delta"] = recent10 - df.level
    df["recent5_vs_previous5"] = recent5 - previous5
    df["has_10_history"] = df.hist_n.ge(10).astype(int)
    q75 = past.transform(lambda s: s.shift(1).expanding(min_periods=MIN_HISTORY).quantile(.75))
    q25 = past.transform(lambda s: s.shift(1).expanding(min_periods=MIN_HISTORY).quantile(.25))
    df["hist_iqr"] = (q75 - q25).fillna(0.0)
    df["hour"], df["weekday"] = df.posted_at.dt.hour, df.posted_at.dt.weekday
    gap = df.groupby("author", sort=False).posted_at.diff().dt.total_seconds() / 86400
    df["days_since_prev"] = gap.fillna(-1.0)
    prior_gap = gap.groupby(df.author, sort=False).transform(lambda s: s.shift(1).rolling(5, min_periods=5).median())
    df["gap_surprise"] = np.log1p(gap) - np.log1p(prior_gap)
    # No median from other authors, current/future posts or held-out data.
    df["age_days"] = (df.crawled_at - df.posted_at).dt.total_seconds() / 86400
    followers = pd.Series(dtype=float) if followers is None else followers
    df["log_followers"] = np.log1p(df.author.map(followers).fillna(0).clip(lower=0))
    for source, output, logarithmic in [("n_chars", "length_delta", True),
                                         ("caps_fraction", "caps_delta", False),
                                         ("n_urls", "urls_delta", True),
                                         ("n_mentions", "mentions_delta", True)]:
        values = np.log1p(df[source]) if logarithmic else df[source]
        median = values.groupby(df.author, sort=False).transform(
            lambda s: s.shift(1).expanding(min_periods=MIN_HISTORY).median())
        df[output] = values - median
    novelty = []
    for _, group in df.groupby("author", sort=False):
        people, organisations, types = set(), set(), Counter()
        for position, row in enumerate(group.itertuples()):
            current_people, current_orgs = set(row.person_codes), set(row.org_codes)
            novelty.append((len(current_people - people),
                            len(current_people & people) / max(1, len(current_people)),
                            len(current_orgs - organisations),
                            len(current_orgs & organisations) / max(1, len(current_orgs)),
                            types[row.content_type] / position if position else np.nan))
            people.update(current_people)
            organisations.update(current_orgs)
            types[row.content_type] += 1
    df[["person_new_count", "person_seen_fraction", "org_new_count", "org_seen_fraction",
        "past_same_type_fraction"]] = pd.DataFrame(novelty, index=df.index)
    return df[KEYS + ["text", "content_text"] + NUMERIC + CATEGORICAL].copy()


def load_experiment_data(data_dir: Path | str):
    """Return labelled rows, all own-post history, numeric groups and an audit report."""
    data_dir = Path(data_dir)
    digest = hashlib.sha256()
    file_count = 0

    def read(path: Path) -> bytes:
        nonlocal file_count
        blob = path.read_bytes()
        name = path.relative_to(data_dir).as_posix().encode()
        digest.update(len(name).to_bytes(8, "big") + name + hashlib.sha256(blob).digest())
        file_count += 1
        return blob

    labels_blob = read(data_dir / "labels.csv")
    labels = pd.read_csv(io.BytesIO(labels_blob))
    split = json.loads(read(data_dir / "split.json"))
    _validate_keys(labels, "labels.csv")
    if not {"label", "split"} <= set(labels):
        raise ValueError("labels.csv requires label and split columns")
    if not labels.label.isin(CLASS_NAMES).all():
        raise ValueError("labels.csv labels must be 0, 1 or 2")
    if "class" in labels and not labels["class"].eq(labels.label.map(CLASS_NAMES)).all():
        raise ValueError("labels.csv class names do not match numeric labels")
    if set(labels.split) != {"train", "test"} or not isinstance(split, dict):
        raise ValueError("labels.csv and split.json must define train and test")
    for name in ("train", "test"):
        members = split.get(name)
        if not isinstance(members, list) or not all(isinstance(a, str) for a in members):
            raise ValueError(f"split.json {name} must be a list of author codes")
        if len(members) != len(set(members)) or set(members) != set(labels.loc[labels.split == name, "author"]):
            raise ValueError(f"split.json {name} author list differs from labels.csv or contains duplicates")
    if set(split["train"]) & set(split["test"]):
        raise ValueError("train and test authors overlap")

    followers = {}
    profiles = sorted((data_dir / "authors").glob("AUTHOR_*.json"))
    feeds = sorted((data_dir / "posts").glob("AUTHOR_*.json"))
    if not profiles or not feeds:
        raise ValueError("authors and posts JSON files are both required")
    for path in profiles:
        profile = json.loads(read(path))
        owner = profile.get("author", path.stem)
        if owner in followers:
            raise ValueError("author profiles contain duplicate author codes")
        followers[owner] = profile.get("followers")
    if set(labels.author) - set(followers):
        raise ValueError("profiles are missing for labelled authors")
    followers = pd.to_numeric(pd.Series(followers), errors="coerce")
    rows = []
    for path in feeds:
        feed = json.loads(read(path))
        owner = feed["author"]
        for post in feed["posts"]:
            if _object(post.get("author")).get("code") == owner:
                rows.append(extract_post(owner, post, feed.get("crawledAt")))
    if not rows:
        raise ValueError("feeds contain no own posts")
    history = pd.DataFrame(rows)
    # Match the release loader's timestamp conversion; missing crawl dates stay NaN.
    history["crawled_at"] = pd.to_datetime(history.crawled_at, utc=True, errors="coerce")
    history = _ordered_history(history)
    rebuilt = relabel(history, split["train"])
    verification = verify_labels(labels, rebuilt, labels_sha256=hashlib.sha256(labels_blob).hexdigest())
    if verification["boundary_roundoff_count"]:
        print(f"Label integrity: {verification['exact_matches']:,} exact matches, "
              f"{verification['boundary_roundoff_count']:,} platform boundary-roundoff differences "
              f"(max distance {verification['max_error']:.17g}); official targets unchanged", flush=True)
    low, high = rebuilt.attrs["cutoffs"]
    stored = split.get("cutoffs_on_log_residual", {})
    if not isinstance(stored, dict) or not {"low", "high"} <= set(stored):
        raise ValueError("split.json must provide low and high residual cutoffs")
    if not np.allclose([low, high], [stored["low"], stored["high"]], rtol=0, atol=5.00001e-7):
        raise ValueError("split.json cutoffs differ from recomputed cutoffs beyond six-decimal rounding")
    built = build_features(history, followers)
    labelled = labels[KEYS + ["split", "label"]].merge(built, on=KEYS, how="left",
                                                      sort=False, validate="one_to_one")
    if labelled.level.isna().any():
        raise ValueError("labelled posts are missing or lack five earlier own posts")
    labelled["label"] = labelled.label.astype(float).where(labelled.split == "train")
    report = {
        "input_sha256": digest.hexdigest(), "input_file_count": file_count,
        "input_hash_policy": "labels.csv, split.json, sorted authors JSON, sorted posts JSON; relative path and file SHA256",
        "counts": {"own_posts": len(history), "own_authors": int(history.author.nunique()),
                   "early_history_posts": int(history.hist_n.lt(MIN_HISTORY).sum()),
                   "train_rows": int(labelled.split.eq("train").sum()),
                   "test_rows": int(labelled.split.eq("test").sum())},
        "label_consistency_checked": True,
        "label_verification": verification,
        "cutoffs": {"low": low, "high": high, "stored_match_six_decimal_rounding": True},
        "missing_numeric": {name: int(labelled[name].isna().sum()) for name in NUMERIC},
        "missing_content_type": int(history.content_type.eq("unknown").sum()),
        "missing_followers_profiles": int(followers.isna().sum()),
        "future_shared_excluded": int(history.future_shared_excluded.sum()),
        "shared_timestamp_missing": int(history.shared_timestamp_missing.sum()),
        "truncated_documents": int(history.document_truncated.sum() + history.shared_document_truncated.sum()),
        "text_policy": {"document_max_chars": DOCUMENT_MAX_CHARS,
                        "document_limit": "per own/shared document transcript before pseudonym normalisation",
                        "sources": TEXT_SOURCES, "subtitle_included": False,
                        "source_markers": True, "empty_text": "",
                        "future_shared_policy": "exclude all nested shared sources when original is future-dated"},
        "feature_policy": {"history": "strictly earlier own posts only; reject missing/tied publication times",
                           "age_missing": "NaN; imputation must be fit on training rows",
                           "person_novelty": "codes scoped to the current author; raw codes never enter feature table",
                           "mention_codes": "allowed text and structured mentions, plus shared company/owner AUTHOR/local PERSON publisher",
                           "emoji": "Unicode code points in U+1F300–1FAFF and U+2600–27BF, not grapheme clusters",
                           "profile_fields": ["followers"], "baseline_text_counts": "original own text before normalisation"},
    }
    return labelled, history, {name: list(columns) for name, columns in FEATURE_GROUPS.items()}, report
