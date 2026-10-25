"""Small, wholly synthetic release; no private dataset rows are bundled."""
import json
from pathlib import Path

import numpy as np

from kairos.data import load_posts
from kairos.label import relabel


def make_release(root: Path):
    (root / "authors").mkdir(parents=True)
    (root / "posts").mkdir()
    (root / "media").mkdir()
    train, test = [], []
    rng = np.random.default_rng(42)
    for i in range(12):
        author = f"AUTHOR_{i:04d}"
        (train if i < 9 else test).append(author)
        posts = []
        for j in range(24):
            posts.append({"urn": f"POST_{i:04d}_{j:04d}", "author": {"code": author},
                "postedDateTimestamp": 1609459200000 + j * 86400000,
                "totalReactionCount": int(np.exp(rng.normal(3 + i / 10, 1.2))),
                "text": f"{'Learn code today' if j % 2 else 'Join our launch'} #{j % 5}\nWhat do you think?",
                "contentType": "image" if j % 3 else "text", "document": {"transcript": "example document"} if j % 7 == 0 else None})
        (root / "posts" / f"{author}.json").write_text(json.dumps({"author": author, "crawledAt": "2026-01-01", "posts": posts}))
        (root / "authors" / f"{author}.json").write_text(json.dumps({"author": author, "followers": 100 + i,
                                                                  "skills": ["python", "writing"]}))
    # Loader expects labels.csv to exist before label reproduction.
    (root / "labels.csv").write_text("author,post,split,label,class\n")
    labels = relabel(load_posts(root), train)
    low, high = labels.attrs["cutoffs"]
    labels["split"] = labels.author.map(lambda x: "train" if x in train else "test")
    labels["class"] = labels.label.map({0: "below", 1: "typical", 2: "above"})
    labels[["author", "post", "split", "label", "class"]].to_csv(root / "labels.csv", index=False)
    (root / "split.json").write_text(json.dumps({"train": train, "test": test,
                                                 "cutoffs_on_log_residual": {"low": low, "high": high}}))
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/night.json").read_text())
    config.update(expected=labels.split.value_counts().to_dict(), folds=3, threads=1, baseline_iterations=5,
                  catboost=[{"depth": 2, "iterations": 8, "learning_rate": .2, "class_power": .5}],
                  tfidf={"word_features": 100, "char_features": 100, "min_df": 1, "C": [1.0]},
                  bootstrap_samples=10, enable_images=False)
    return config
