"""Train-only exploration: coverage, error slices, label support, candidate features.

No original text, author names, media or current outcome counters are emitted.
Current reactions and residuals are used only in the aggregate diagnostic report.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TQDM_DISABLE", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score

from kairos.data import load_labels, load_posts, load_split
from kairos.label import author_level, fit_cutoffs
from student_features import build_features, normalize_text

OUT = ROOT / "results" / "exploration_v2"


def text_value(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(text_value(v) for v in value)
    if isinstance(value, dict):
        return " ".join(text_value(value[k]) for k in ("text", "transcript", "title", "content") if k in value)
    return ""


def as_dict(value):
    return value if isinstance(value, dict) else {}


def writing_system(text):
    text = re.sub(r"⟨[^⟩]+⟩", "", text)
    latin = len(re.findall("[A-Za-z]", text))
    cyrillic = len(re.findall("[А-Яа-яЁё]", text))
    if latin + cyrillic < 4:
        return "other_or_empty"
    if cyrillic / (latin + cyrillic) > .8:
        return "mostly_cyrillic"
    if latin / (latin + cyrillic) > .8:
        return "mostly_latin"
    return "mixed"


def extra_content(authors):
    rows = []
    for author in sorted(authors):
        feed = json.loads((ROOT / "data" / "posts" / f"{author}.json").read_text(encoding="utf-8"))
        for post in feed["posts"]:
            if as_dict(post.get("author")).get("code") != author:
                continue
            article = as_dict(post.get("article"))
            document = as_dict(post.get("document"))
            reshared = as_dict(post.get("resharedPost"))
            shared_article = as_dict(reshared.get("article"))
            entity = as_dict(post.get("entity"))
            celebration = as_dict(post.get("celebration"))
            transcript = text_value(document.get("transcript"))
            original = text_value(post.get("text"))
            more = [article.get("title"), article.get("subtitle"), document.get("title"),
                    transcript, reshared.get("text"), shared_article.get("title"),
                    entity.get("title"), entity.get("subtitle"), celebration.get("text")]
            expanded = "\n".join([original] + [text_value(t) for t in more if text_value(t)])
            images = post.get("images")
            rows.append(dict(author=author, post=post["urn"], expanded_text=normalize_text(expanded),
                has_extra_text=int(len(expanded) > len(original)), has_transcript=int(bool(transcript)),
                has_reshared=int(bool(reshared)), has_article=int(bool(article)),
                has_document=int(bool(document)), has_video=int(bool(post.get("video"))),
                image_count=len(images) if isinstance(images, list) else int(bool(post.get("image"))),
                document_pages=float(document.get("totalPageCount") or 0),
                is_brand_partnership=int(bool(post.get("isBrandPartnership"))),
                writing_system=writing_system(original),
                original_chars=len(original), extra_chars=len(expanded) - len(original)))
    return pd.DataFrame(rows)


def lag_features(posts):
    rows = []
    posts = posts.sort_values(["author", "posted_at", "post"], kind="stable")
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    for _, group in posts.groupby("author", sort=False):
        seconds = (group.posted_at - epoch).dt.total_seconds().to_numpy()
        values = np.log1p(group.reactions.to_numpy(dtype=float))
        types = group.content_type.to_numpy()
        for i, (_, record) in enumerate(group.iterrows()):
            end = int(np.searchsorted(seconds, seconds[i], side="left"))
            past = values[:end]
            row = dict(author=record.author, post=record.post,
                       timestamp_tie=int(np.count_nonzero(seconds == seconds[i]) > 1))
            if end:
                level = float(np.median(past))
                row["last_reaction_delta"] = float(past[-1] - level)
                for window in (3, 10, 20):
                    recent = past[-window:]
                    row[f"lag{window}_median_delta"] = float(np.median(recent) - level)
                    row[f"lag{window}_mean_delta"] = float(recent.mean() - level)
                    row[f"lag{window}_zero_share"] = float(np.mean(recent == 0))
                for half_life in (30, 180):
                    weights = np.exp2(np.maximum(-(seconds[i] - seconds[:end]) / (86400 * half_life), -1000))
                    row[f"ewm_{half_life}d_delta"] = float(np.average(past, weights=weights) - level)
                same = past[types[:end] == types[i]]
                row["type_history_n"] = len(same)
                row["type_level_delta"] = float((same.sum() + 5 * level) / (len(same) + 5) - level)
                row["lag_gap_ratio"] = float((seconds[i] - seconds[end - 1]) / max(86400, np.median(np.diff(seconds[:end])))) if end > 1 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def describe_groups(table, column):
    rows = []
    for value, group in table.groupby(column, observed=True, dropna=False):
        if len(group) < 100:
            continue
        rows.append(dict(group=str(value), n=len(group), authors=group.author.nunique(),
            class_shares=[round(float((group.label == k).mean()), 4) for k in range(3)],
            oof_macro_f1=round(float(f1_score(group.label, group.fusion, average="macro", labels=[0, 1, 2], zero_division=0)), 4),
            error_rate=round(float((group.label != group.fusion).mean()), 4)))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    labels = labels.loc[labels.split == "train"].copy()
    authors = set(labels.author)
    posts = load_posts()
    posts = posts.loc[posts.author.isin(authors)].copy()
    official = author_level(posts)
    low, high = fit_cutoffs(official.resid)
    print("Building train-only table and content coverage...", flush=True)
    base, numeric = build_features(posts, low=low, high=high)
    extra = extra_content(authors)
    lagged = lag_features(posts)
    table = labels.merge(base, on=["author", "post"], validate="one_to_one")
    table = table.merge(extra, on=["author", "post"], validate="one_to_one")
    table = table.merge(lagged, on=["author", "post"], validate="one_to_one")
    extra_numeric = [c for c in extra if c not in ["author", "post", "expanded_text", "writing_system"]]
    extra_numeric += [c for c in lagged if c not in ["author", "post"]]
    joblib.dump(dict(table=table, numeric=numeric, extra_numeric=extra_numeric,
                    cutoffs=[low, high]), OUT / "train_table.joblib")
    official = official[["author", "post", "level", "resid", "reactions", "posted_at"]].rename(columns={"level": "official_level"})
    diagnostics = table.merge(official, on=["author", "post"], validate="one_to_one")
    oof = pd.read_csv(ROOT / "results" / "maria" / "oof_predictions.csv")
    diagnostics = diagnostics.merge(oof[["author", "post", "fusion", "history", "tfidf"]], on=["author", "post"], validate="one_to_one")
    diagnostics["history_bin"] = pd.cut(diagnostics.hist_n, [-1, 9, 19, 49, np.inf], labels=["0-9", "10-19", "20-49", "50+"])
    diagnostics["level_bin"] = pd.cut(diagnostics.official_level, [-1, -low, np.log1p(5), np.log1p(20), np.log1p(100), np.inf], labels=["below impossible", "up to 5 reactions", "5-20", "20-100", "100+"])
    diagnostics["length_bin"] = pd.cut(diagnostics.original_chars, [-1, 0, 150, 500, 1500, np.inf], labels=["empty", "1-150", "151-500", "501-1500", "1500+"])
    diagnostics["year"] = diagnostics.posted_at.dt.year
    digest = diagnostics.text.map(lambda t: hashlib.sha256(re.sub(r"\s+", " ", t.strip().lower()).encode()).hexdigest())
    duplicate_groups = diagnostics.assign(digest=digest).loc[diagnostics.original_chars >= 50].groupby("digest").author.nunique()
    duplicates = set(duplicate_groups[duplicate_groups > 1].index)
    support = diagnostics.official_level <= -low
    distance = np.minimum(abs(diagnostics.resid - low), abs(diagnostics.resid - high))
    sample = table.sample(min(18000, len(table)), random_state=20261025)
    columns = numeric + extra_numeric
    matrix = sample[columns].replace([np.inf, -np.inf], np.nan)
    matrix = matrix.fillna(matrix.median()).fillna(0)
    print("Screening feature associations on a train subsample...", flush=True)
    mi = mutual_info_classif(matrix, sample.label, random_state=20261025, n_jobs=4)
    report = dict(n_train=len(table), authors=len(authors), zero_reactions_share=float((diagnostics.reactions == 0).mean()),
        coverage={c:dict(n=int((diagnostics[c] > 0).sum()), share=float((diagnostics[c] > 0).mean()))
                  for c in ["has_extra_text", "has_transcript", "has_reshared", "has_article", "has_document", "has_video", "image_count", "timestamp_tie"]},
        below_impossible=dict(n=int(support.sum()), actual_below=int(((diagnostics.label == 0) & support).sum()),
                              predicted_below=int(((diagnostics.fusion == 0) & support).sum())),
        near_boundary={str(width):float((distance <= width).mean()) for width in (.01, .05, .1)},
        cross_author_duplicate_text_posts=int((digest.isin(duplicates) & (diagnostics.original_chars >= 50)).sum()),
        slices={c:describe_groups(diagnostics, c) for c in ["content_type", "history_bin", "level_bin", "length_bin", "writing_system", "year", "has_extra_text", "has_reshared", "has_document"]},
        feature_association_screen=sorted([dict(feature=c, mutual_information=float(v)) for c,v in zip(columns,mi)], key=lambda x:-x["mutual_information"]),
        interpretation="Exploratory train-only associations, not causal effects or held-out evidence of improvement.")
    (OUT / "diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ["slices", "feature_association_screen"]}, indent=2), flush=True)
    print("Top associations:", report["feature_association_screen"][:12], flush=True)


if __name__ == "__main__":
    main()
