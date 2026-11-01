from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

from kairos.metrics import score
from lab1_features import CATEGORICAL
from train_lab1 import SEED, blend, boosted, checked_table, select_blend, tabular, text_fingerprint


def main():
    ap = argparse.ArgumentParser(description="Compare boosting depths on training-author validation only")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--previous", type=Path, default=Path("results/validation-timestamps"))
    ap.add_argument("--output", type=Path, default=Path("results/search"))
    ap.add_argument("--depths", type=int, nargs="+", default=[8, 10])
    ap.add_argument("--iterations", type=int, default=2200)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    table, numeric = checked_table(args.data)
    train = table[table.split == "train"].reset_index(drop=True)
    fit_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED)
                            .split(train, groups=train.author))
    fit, val = train.iloc[fit_idx], train.iloc[val_idx]
    previous = np.load(args.previous / "validation_probabilities.npz")
    previous_config = json.loads((args.previous / "selection.json").read_text())
    if not np.array_equal(previous["authors"], val.author) or not np.array_equal(previous["labels"], val.label):
        raise SystemExit("Previous text probabilities belong to another validation partition")
    if (previous_config.get("text_input_sha256") != text_fingerprint(train, numeric)
            or previous_config.get("text_c", 1.0) != 1.0
            or "posts" not in previous.files or not np.array_equal(previous["posts"], val.post)):
        raise SystemExit("Previous text probabilities have different input features, keys or classifier settings")
    text_prob = previous["text"]
    trials = []
    with threadpool_limits(limits=args.threads):
        for depth in args.depths:
            started = time.time()
            model = boosted(args.iterations, args.threads, depth)
            model.fit(tabular(fit, numeric), fit.label, cat_features=CATEGORICAL,
                      eval_set=(tabular(val, numeric), val.label),
                      early_stopping_rounds=150, use_best_model=True, verbose=200)
            prob = model.predict_proba(tabular(val, numeric))
            config = select_blend(val.label, prob, text_prob)
            trial = {"depth": depth, "max_iterations": args.iterations,
                     "selected_iterations": model.tree_count_, "text_c": 1.0,
                     "validation_posts": len(val), "validation_authors": int(val.author.nunique()),
                     "tree": score(val.label, prob.argmax(axis=1)),
                     "blend": score(val.label, blend(prob, text_prob, config)),
                     "blend_config": config, "seconds": round(time.time() - started, 1)}
            trials.append(trial)
            (args.output / f"depth-{depth}.json").write_text(json.dumps(trial, indent=2))
            np.savez_compressed(args.output / f"depth-{depth}.npz", tree=prob, text=text_prob,
                                labels=val.label.to_numpy(), authors=val.author.to_numpy(dtype=str),
                                posts=val.post.to_numpy(dtype=str))
            (args.output / "trials.json").write_text(json.dumps(trials, indent=2))
            print(json.dumps(trial, indent=2), flush=True)
            del model


if __name__ == "__main__":
    main()
