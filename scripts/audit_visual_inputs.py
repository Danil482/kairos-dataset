"""Audit image availability and cross-author reuse without reading image contents."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
from scripts.semantic_embeddings import SOURCE, save_json
from scripts.visual_embeddings import OUT


def overlap_by_fold(table, index):
    by_post = defaultdict(set)
    for pair in index["pairs"]:
        by_post[(pair["author"], pair["post"])].add(pair["image_index"])
    flags, counts = {}, {}
    for number in (1, 2, 3):
        plan = json.loads((ROOT / "results" / "maria" / f"fold_{number}.json").read_text())
        fitting_authors = set(plan["fitting_authors"])
        validation_authors = set(plan["validation_authors"])
        known = {number for (author, _), images in by_post.items() if author in fitting_authors for number in images}
        n_images, n_overlap = 0, 0
        for row in table[table.author.isin(validation_authors)].itertuples():
            key = (row.author, row.post)
            images = by_post.get(key, set())
            flags[key] = bool(images.intersection(known))
            n_images += bool(images)
            n_overlap += flags[key]
        counts[str(number)] = dict(validation_posts_with_images=n_images, with_any_image_seen_on_fitting_authors=n_overlap)
    return flags, counts


def main():
    table = joblib.load(SOURCE)["table"]
    index = json.loads((OUT / "media_index.json").read_text())
    by_image, by_kind = defaultdict(set), Counter()
    post_keys = set()
    for pair in index["pairs"]:
        by_image[pair["image_index"]].add(pair["author"])
        by_kind.update(pair["kinds"])
        post_keys.add((pair["author"], pair["post"]))
    empty_with_images = sum(row.original_chars == 0 and (row.author, row.post) in post_keys for row in table.itertuples())
    _, counts = overlap_by_fold(table, index)
    report = dict(train_posts=len(table), content_images=len(index["files"]), unique_post_image_pairs=len(index["pairs"]),
        posts_with_images=len(post_keys), empty_original_with_images=empty_with_images,
        images_used_by_multiple_authors=sum(len(authors)>1 for authors in by_image.values()),
        pair_counts_by_kind=dict(by_kind), cross_author_image_overlap=counts,
        note="Identical file paths only; perceptually similar but differently encoded images are not detected")
    save_json(OUT / "input_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
