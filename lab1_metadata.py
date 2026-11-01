from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROFILE_NUM = ["profile_skills", "profile_languages", "profile_positions", "profile_educations",
               "profile_honors", "profile_publications", "profile_premium", "profile_summary_chars",
               "profile_headline_chars", "profile_first_career_year"]
POST_NUM = ["media_images", "media_videos", "image_mean_width", "image_mean_height", "image_max_area",
            "video_duration_raw", "document_pages", "document_text_chars", "has_article", "has_poll",
            "is_reshare", "brand_partnership", "structured_mentions", "structured_company_mentions",
            "log_media_bytes", "media_files", "reshared_media_files"]
PROFILE_CAT = ["profile_country", "profile_language", "profile_industry", "profile_staff_range"]


def as_list(value):
    return value if isinstance(value, list) else []


def as_dict(value):
    return value if isinstance(value, dict) else {}


def text(value):
    return value if isinstance(value, str) else ""


def transcript(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(transcript(v) for v in value)
    if isinstance(value, dict):
        return " ".join(text(value.get(k)) for k in ["text", "transcript", "content"])
    return ""


def enrich_posts(posts, data):
    data = Path(data)
    profiles = []
    for path in sorted((data / "authors").glob("AUTHOR_*.json")):
        p = json.loads(path.read_text(encoding="utf-8"))
        positions = as_list(p.get("fullPositions")) or as_list(p.get("position"))
        current = as_dict(positions[0]) if positions else {}
        years = [as_dict(as_dict(v).get("start")).get("year") for v in positions]
        years = [v for v in years if isinstance(v, int) and 1900 < v < 2100]
        profile_text = " ".join([
            text(p.get("headline"))[:300], text(p.get("summary"))[:1500],
            " ".join(text(as_dict(v).get("name")) for v in as_list(p.get("skills")))[:1000],
            " ".join(text(as_dict(v).get("title")) for v in positions[:8])[:1000],
        ])
        profiles.append({
            "author": p.get("author", path.stem), "profile_text": profile_text or "EMPTY_PROFILE",
            "profile_skills": len(as_list(p.get("skills"))),
            "profile_languages": len(as_list(p.get("languages"))),
            "profile_positions": len(positions),
            "profile_educations": len(as_list(p.get("educations"))),
            "profile_honors": len(as_list(p.get("honors"))),
            "profile_publications": len(as_list(p.get("publications"))),
            "profile_premium": int(bool(p.get("isPremium"))),
            "profile_summary_chars": len(text(p.get("summary"))),
            "profile_headline_chars": len(text(p.get("headline"))),
            "profile_first_career_year": min(years) if years else np.nan,
            "profile_country": text(as_dict(p.get("geo")).get("countryCode")) or "unknown",
            "profile_language": text(as_dict(p.get("primaryLocale")).get("language")) or "unknown",
            "profile_industry": text(current.get("companyIndustry")) or "unknown",
            "profile_staff_range": text(current.get("companyStaffCountRange")) or "unknown",
        })
    rows = []
    for path in sorted((data / "posts").glob("AUTHOR_*.json")):
        feed = json.loads(path.read_text(encoding="utf-8"))
        owner = feed["author"]
        for p in feed["posts"]:
            if as_dict(p.get("author")).get("code") != owner:
                continue
            images = as_list(p.get("images")) or as_list(p.get("image"))
            images = [as_dict(v) for v in images]
            videos = [as_dict(v) for v in as_list(p.get("video"))]
            doc = as_dict(p.get("document"))
            article = as_dict(p.get("article"))
            doc_text = transcript(doc.get("transcript"))[:20000]
            extra = " ".join([text(article.get("title")), text(article.get("subtitle")),
                              doc_text, text(as_dict(p.get("resharedPost")).get("text"))[:4000]])
            widths = [v.get("width") for v in images if isinstance(v.get("width"), (int, float))]
            heights = [v.get("height") for v in images if isinstance(v.get("height"), (int, float))]
            rows.append({
                "author": owner, "post": p["urn"], "extra_text": extra,
                "release_crawled_at": feed.get("crawledAt"),
                "media_images": len(images), "media_videos": len(videos),
                "image_mean_width": float(np.mean(widths)) if widths else 0,
                "image_mean_height": float(np.mean(heights)) if heights else 0,
                "image_max_area": max([float(v.get("width") or 0) * float(v.get("height") or 0)
                                       for v in images], default=0),
                "video_duration_raw": max([float(v.get("duration") or 0) for v in videos], default=0),
                "document_pages": float(doc.get("totalPageCount") or 0),
                "document_text_chars": len(doc_text), "has_article": int(bool(article)),
                "has_poll": int(bool(p.get("poll"))), "is_reshare": int(bool(p.get("resharedPost"))),
                "brand_partnership": int(bool(p.get("isBrandPartnership"))),
                "structured_mentions": len(as_list(p.get("mentions"))),
                "structured_company_mentions": len(as_list(p.get("companyMentions"))),
            })
    media = {}
    manifest = data / "media" / "manifest.jsonl"
    if manifest.is_file():
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                entry = json.loads(line)
                key = (entry.get("author"), entry.get("post"))
                counts = media.setdefault(key, [0, 0, 0])
                counts[0] += int(entry.get("bytes") or 0)
                counts[1] += 1
                counts[2] += int(bool(entry.get("from_reshared_post")))
    for row in rows:
        size, files, shared = media.get((row["author"], row["post"]), [0, 0, 0])
        row.update(log_media_bytes=float(np.log1p(size)), media_files=files, reshared_media_files=shared)
    result = posts.merge(pd.DataFrame(rows), on=["author", "post"], how="left", validate="one_to_one").merge(
        pd.DataFrame(profiles), on="author", how="left", validate="many_to_one")
    result["crawled_at"] = pd.to_datetime(result.pop("release_crawled_at"), utc=True, format="mixed", errors="coerce")
    return result
