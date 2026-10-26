"""Aggregate content-media coverage and within-author associations on train."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "exploration_v2"
CONTENT_KINDS = {"post_image", "article_image", "post_video", "video_poster",
                 "document_cover", "document_page", "post_document", "celebration_image"}


def main():
    table = joblib.load(OUT / "train_table.joblib")["table"]
    keys = set(zip(table.author, table.post))
    kinds, positions = Counter(), set()
    posts_by_mime, unique_by_mime = defaultdict(set), defaultdict(set)
    ignored = Counter()
    for line in (ROOT / "data" / "media" / "manifest.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        key = (row.get("author"), row.get("post"))
        if key not in keys:
            continue
        kind = row.get("kind")
        kinds[kind] += 1
        if kind not in CONTENT_KINDS:
            ignored[kind] += 1
            continue
        mime = (row.get("content_type") or "unknown").split("/")[0]
        posts_by_mime[mime].add(key)
        unique_by_mime[mime].add(row.get("file"))
        positions.add((key, row.get("file")))
    table = table.assign(above=table.label.eq(2), below=table.label.eq(0))
    pairs = []
    for _, group in table.groupby("author"):
        original = group[group.has_reshared.eq(0)]
        shared = group[group.has_reshared.eq(1)]
        if len(original) >= 3 and len(shared) >= 3:
            pairs.append([shared.above.mean() - original.above.mean(),
                          shared.below.mean() - original.below.mean()])
    pairs = np.asarray(pairs)
    rng = np.random.default_rng(42)
    boot = np.array([pairs[rng.integers(0, len(pairs), len(pairs))].mean(axis=0) for _ in range(1000)])
    report = dict(media_kinds_raw_rows=dict(kinds), excluded_identity_media_rows=dict(ignored),
        unique_content_post_file_pairs=len(positions),
        posts_with_content_media_by_mime={k:len(v) for k,v in posts_by_mime.items()},
        unique_content_files_by_mime={k:len(v) for k,v in unique_by_mime.items()},
        paired_reshare=dict(authors=len(pairs), mean_above_difference=float(pairs[:, 0].mean()),
            mean_below_difference=float(pairs[:, 1].mean()),
            bootstrap_95=np.quantile(boot, [.025, .975], axis=0).tolist()),
        interpretation="Author-equal associations; other differences such as timing and topics remain uncontrolled.")
    (OUT / "media_and_within_author.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
