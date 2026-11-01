"""Validate the release and export compact tables for text experiments.

    python prepare_text.py --data data --output data/prepared

Only train.csv.gz contains labels. The official author split and the baseline's
numeric features are preserved; text pseudonyms are reduced to generic tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from baseline import NUMERIC
from kairos import features
from kairos.data import DATA_DIR, load_followers, load_labels, load_posts, load_split
from kairos.label import CLASS_NAMES, author_level, relabel

KEYS = ["author", "post"]
EXPORT_COLUMNS = KEYS + ["text"] + NUMERIC + features.CONTENT_CAT
PSEUDONYM = re.compile(r"⟨(PERSON|ORG|AUTHOR)_[^⟩]+⟩")


def normalize_text(text: str) -> str:
    text = PSEUDONYM.sub(r"⟨\1⟩", text).strip()
    return text or "__empty__"


def validate_keys(frame: pd.DataFrame, name: str) -> None:
    if frame[KEYS].isna().any().any():
        raise ValueError(f"{name} contains a missing author or post key")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{name} contains duplicate (author, post) keys")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(data: Path, output: Path) -> dict:
    labels = load_labels(data)
    required = set(KEYS + ["split", "label"])
    if not required <= set(labels.columns):
        raise ValueError(f"labels.csv is missing columns: {sorted(required - set(labels.columns))}")
    validate_keys(labels, "labels.csv")
    if not labels["label"].isin(CLASS_NAMES).all():
        raise ValueError("labels.csv labels must be 0, 1 or 2, with no missing values")
    if set(labels["split"]) != {"train", "test"}:
        raise ValueError("labels.csv must contain both train and test, with no other split values")

    authors = {name: set(labels.loc[labels["split"] == name, "author"])
               for name in ("train", "test")}
    if authors["train"] & authors["test"]:
        raise ValueError("labels.csv has authors present in both train and test")
    if not (data / "split.json").is_file():
        raise ValueError("split.json is required to verify the official split and labels")
    split = load_split(data)
    for name in ("train", "test"):
        members = split.get(name)
        if not isinstance(members, list) or not all(isinstance(a, str) for a in members):
            raise ValueError(f"split.json must contain a list of author codes in {name}")
        if len(members) != len(set(members)):
            raise ValueError(f"split.json contains duplicate authors in {name}")
        if not authors[name] <= set(members):
            raise ValueError(f"labels.csv has {name} authors absent from split.json {name}")
    if set(split["train"]) & set(split["test"]):
        raise ValueError("split.json has authors present in both train and test")

    if not any((data / "authors").glob("AUTHOR_*.json")):
        raise ValueError("authors/AUTHOR_*.json profiles are required for baseline follower features")
    followers = load_followers(data)
    missing_profiles = set(labels["author"]) - set(followers.index)
    if missing_profiles:
        raise ValueError(f"author profiles are missing for {len(missing_profiles)} labelled authors")
    if not any((data / "posts").glob("AUTHOR_*.json")):
        raise ValueError("posts/AUTHOR_*.json feeds are required")
    posts = load_posts(data)
    validate_keys(posts, "own posts")
    levelled = author_level(posts)
    built = features.build(levelled, followers)
    # Upstream fills age with a median across the whole release. Keep missing
    # ages here so held-out authors cannot affect training features. HGB handles
    # NaN; the embedding classifier fits its imputer on training rows only.
    built["age_days"] = (built["crawled_at"] - built["posted_at"]).dt.total_seconds() / 86400
    table = labels.merge(built, on=KEYS, how="left",
                         validate="one_to_one")
    if table["level"].isna().any():
        raise ValueError("some labelled posts are missing from the feeds or have fewer than five earlier posts")

    recomputed = relabel(posts, split["train"])
    both = labels.merge(recomputed[KEYS + ["label"]], on=KEYS, how="outer",
                        suffixes=("_released", "_recomputed"), validate="one_to_one")
    if not both["label_released"].eq(both["label_recomputed"]).all():
        raise ValueError("labels.csv differs from the labels recomputed from posts and split.json")

    table["text"] = table["text"].map(normalize_text)
    manifest = {"format_version": 1, "label_consistency_checked": True, "files": {}}
    output.mkdir(parents=True, exist_ok=True)
    for name in ("train", "test"):
        columns = EXPORT_COLUMNS + (["label"] if name == "train" else [])
        exported = table.loc[table["split"] == name, columns].copy()
        if name == "train":
            exported["label"] = exported["label"].astype(int)
        path = output / f"{name}.csv.gz"
        exported.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
        manifest["files"][name] = {
            "file": path.name,
            "sha256": sha256(path),
            "rows": len(exported),
            "authors": int(exported["author"].nunique()),
            "columns": columns,
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                           encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR, help="unpacked release (default: data/)")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "prepared",
                        help="export directory (default: data/prepared/)")
    args = parser.parse_args()
    try:
        manifest = prepare(args.data, args.output)
    except ValueError as exc:
        parser.exit(1, f"error: {exc}\n")
    for name, info in manifest["files"].items():
        print(f"{name}: {info['rows']:,} posts / {info['authors']:,} authors -> "
              f"{args.output / info['file']}")
    print(f"Labels verified; wrote {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
