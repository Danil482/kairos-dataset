"""Read the Kairos release from disk and expose lightweight metadata features."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

PROFILE_COLUMNS = [
    "profile_n_skills",
    "profile_has_skills",
    "profile_headline_chars",
    "profile_about_chars",
    "profile_n_positions",
]
MEDIA_KINDS = ("image", "video", "document", "article")

# Media-manifest fields, and file extensions per media kind.
_TYPE_KEYS = (
    "media_type",
    "mediaType",
    "mime_type",
    "mimeType",
    "kind",
    "type",
    "extension",
    "ext",
)
_PATH_KEYS = (
    "path",
    "relative_path",
    "media_path",
    "filename",
    "file",
    "name",
    "sha256",
)
_IMAGE_EXT = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".bmp",
    ".heic",
    ".tif",
    ".tiff",
}
_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg"}
_DOCUMENT_EXT = {
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".txt",
    ".rtf",
}


# --------------------------------------------------------------------------- #
# Release files
# --------------------------------------------------------------------------- #
def _release(data_dir: Path | str) -> Path:
    data_dir = Path(data_dir)
    if not (data_dir / "posts").is_dir() or not (data_dir / "labels.csv").is_file():
        raise SystemExit(
            f"No release in {data_dir}. Request the password and download the core "
            "archive at https://kairos-labs.ai/dataset/, then unpack it there:\n"
            f"    7z x lab1_core.7z.001 -o{data_dir}"
        )
    return data_dir


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_profiles(data_dir: Path) -> Iterator[tuple[str, dict]]:
    """Yield (author, profile) for every authors/AUTHOR_*.json file."""
    for file in sorted((data_dir / "authors").glob("AUTHOR_*.json")):
        profile = _read_json(file)
        yield profile.get("author", file.stem), profile


def load_posts(data_dir: Path | str = DATA_DIR, own_only: bool = True) -> pd.DataFrame:
    """One row per post, in file order.

    A feed contains the author's own posts and reposts. By default this returns
    only the author's own posts, which are the population on which labels are defined.
    """
    data_dir = _release(data_dir)
    rows = []
    for file in tqdm(
        sorted((data_dir / "posts").glob("AUTHOR_*.json")), desc="posts", unit="file"
    ):
        feed = _read_json(file)
        owner = feed["author"]
        for post in feed["posts"]:
            post_author = post.get("author")
            own = isinstance(post_author, dict) and post_author.get("code") == owner
            if own_only and not own:
                continue
            rows.append(
                {
                    "author": owner,
                    "post": post["urn"],
                    "own": own,
                    "posted_at": post.get("postedDateTimestamp"),
                    "crawled_at": feed.get("crawledAt"),
                    "reactions": int(post.get("totalReactionCount") or 0),
                    "text": post.get("text") or "",
                    "content_type": post.get("contentType") or "unknown",
                }
            )
    df = pd.DataFrame(rows)
    df["posted_at"] = pd.to_datetime(df["posted_at"], unit="ms", utc=True)
    df["crawled_at"] = pd.to_datetime(df["crawled_at"], utc=True, errors="coerce")
    return df


def load_followers(data_dir: Path | str = DATA_DIR) -> pd.Series:
    """Follower count per author (a 2026 snapshot, not historical audience size)."""
    data_dir = _release(data_dir)
    counts = {
        author: profile.get("followers") for author, profile in _iter_profiles(data_dir)
    }
    return pd.to_numeric(pd.Series(counts, name="followers"), errors="coerce")


def load_labels(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """author, post, split (train/test), label (0/1/2), class."""
    return pd.read_csv(_release(data_dir) / "labels.csv")


def load_split(data_dir: Path | str = DATA_DIR) -> dict:
    return _read_json(_release(data_dir) / "split.json")


# --------------------------------------------------------------------------- #
# Author profile features
# --------------------------------------------------------------------------- #
def _list_size(value: Any) -> float:
    """Length of a list field; NaN when the field is absent or not a list."""
    return float(len(value)) if isinstance(value, list) else float("nan")


def _text_length(value: Any) -> float:
    """Stripped character count of a string field; NaN when absent."""
    return float(len(value.strip())) if isinstance(value, str) else float("nan")


def _profile_row(author: str, profile: dict) -> dict:
    n_skills = _list_size(profile.get("skills"))
    return {
        "author": author,
        "profile_n_skills": n_skills,
        "profile_has_skills": (
            float(n_skills > 0) if pd.notna(n_skills) else float("nan")
        ),
        "profile_headline_chars": _text_length(profile.get("headline")),
        "profile_about_chars": _text_length(profile.get("summary")),
        "profile_n_positions": _list_size(profile.get("fullPositions")),
    }


def load_author_skills(data_dir: Path | str = DATA_DIR) -> dict[str, list[str]]:
    """Author -> list of skill names (empty list when the profile has none)."""
    data_dir = _release(data_dir)
    skills = {}
    for author, profile in _iter_profiles(data_dir):
        entries = profile.get("skills") or []
        names = [e.get("name") if isinstance(e, dict) else e for e in entries]
        skills[author] = [
            name.strip() for name in names if isinstance(name, str) and name.strip()
        ]
    return skills


def load_author_features(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Small profile summaries from authors/*.json, indexed by author.

    Reads the top-level ``skills``, ``headline``, ``summary`` and ``fullPositions``
    fields. Designed for ``features.build(..., profile_features=...)``. A missing
    field stays NaN; it is not treated as a confirmed zero.
    """
    data_dir = _release(data_dir)
    rows = [
        _profile_row(author, profile) for author, profile in _iter_profiles(data_dir)
    ]
    return (
        pd.DataFrame(rows, columns=["author", *PROFILE_COLUMNS])
        .drop_duplicates("author", keep="last")
        .set_index("author")
    )


# --------------------------------------------------------------------------- #
# Media manifest features
# --------------------------------------------------------------------------- #
def _manifest_path(record: dict[str, Any]) -> str:
    """The first path-like field with a file extension, else the last non-empty one."""
    path = ""
    for key in _PATH_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value:
            path = value
            if Path(value).suffix:
                break
    return path


def _manifest_kind(record: dict[str, Any]) -> str:
    """Infer the media kind (image/video/document/article/other) from one manifest row."""
    path = _manifest_path(record)
    type_values = [
        str(record[key]).casefold() for key in _TYPE_KEYS if record.get(key) is not None
    ]
    text = " ".join([*type_values, path.casefold()])
    suffix = Path(path).suffix.casefold()

    if suffix in _IMAGE_EXT or "image/" in text or "photo" in text:
        return "image"
    if suffix in _VIDEO_EXT or "video" in text:
        return "video"
    if suffix in _DOCUMENT_EXT or any(
        word in text for word in ("document", "application/pdf", "presentation")
    ):
        return "document"
    if "article" in text:
        return "article"
    return "other"


def _is_true(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "1", "yes"}
    return bool(value)


def load_media_features(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Per-post media counts and type flags, one row per (author, post).

    Files that belong to a reshared original (``from_reshared_post``) are skipped:
    they do not describe the author's own post.
    """
    data_dir = _release(data_dir)
    columns = ["author", "post", "n_media_items", "has_media"] + [
        f"has_{kind}" for kind in MEDIA_KINDS
    ]
    manifest = data_dir / "media" / "manifest.jsonl"
    if not manifest.is_file():
        return pd.DataFrame(columns=columns)

    rows = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in filter(str.strip, handle):
            record = json.loads(line)
            if record.get("author") is None or record.get("post") is None:
                continue
            if _is_true(record.get("from_reshared_post", False)):
                continue
            rows.append(
                {
                    "author": record["author"],
                    "post": record["post"],
                    "kind": _manifest_kind(record),
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)

    media = pd.DataFrame(rows)
    for kind in MEDIA_KINDS:
        media[f"has_{kind}"] = media["kind"] == kind

    result = media.groupby(["author", "post"], as_index=False).agg(
        n_media_items=("kind", "size"),
        **{f"has_{kind}": (f"has_{kind}", "max") for kind in MEDIA_KINDS},
    )
    result[[f"has_{kind}" for kind in MEDIA_KINDS]] = result[
        [f"has_{kind}" for kind in MEDIA_KINDS]
    ].astype(int)
    result["has_media"] = 1
    return result[columns]
