"""Prepare fit-only pairs and assess a bounded supervised encoder adaptation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from lab_round3 import training_data, probabilities, write_json
from kairos.metrics import score

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--stage", choices=["prepare", "assess"], required=True)
parser.add_argument(
    "--full",
    action="store_true",
    help="Prepare full-train pairs only after an adapted model wins selection.",
)
args = parser.parse_args()
root = Path("results/adaptation") / ("full" if args.full else f"seed{args.seed}")
root.mkdir(parents=True, exist_ok=True)
table, _, config = training_data()
if args.full:
    assert args.stage == "prepare"
    selection = json.loads(Path("results/adaptation/selection.json").read_text())
    assert selection["selected"].startswith("adapted_")
    validation_authors = []
else:
    _, validation_authors = train_test_split(
        sorted(table.author.unique()), test_size=0.2, random_state=args.seed
    )
valid = table.author.isin(validation_authors).to_numpy()
fit = ~valid
assert set(table.loc[fit, "author"]).isdisjoint(table.loc[valid, "author"])

if args.stage == "prepare":
    pool = table.loc[fit & table.text.str.strip().ne("")].drop_duplicates("text")
    sample = pd.concat(
        [
            pool[pool.label == label].sample(n=128, random_state=1234 + label)
            for label in [0, 1, 2]
        ],
        ignore_index=True,
    )
    assert set(sample.author).isdisjoint(validation_authors)
    rng = np.random.default_rng(1234)
    pairs = []
    for i, row in sample.iterrows():
        positive = sample.index[
            (sample.label == row.label) & (sample.index != i)
        ].to_numpy()
        negative = sample.index[sample.label != row.label].to_numpy()
        for target, other in [
            (1, int(rng.choice(positive))),
            (0, int(rng.choice(negative))),
        ]:
            pairs.append(
                {
                    "left": row.text,
                    "right": sample.loc[other, "text"],
                    "target": target,
                    "left_author": row.author,
                    "left_post": row.post,
                    "right_author": sample.loc[other, "author"],
                    "right_post": sample.loc[other, "post"],
                }
            )
    rng.shuffle(pairs)
    assert len(pairs) == 768
    (root / "training_pairs.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs)
    )
    sample[["author", "post", "label"]].to_csv(
        root / "training_sample.csv", index=False
    )
    meta = json.loads(Path("results/fusion/embedding_metadata.json").read_text())
    write_json(
        root / "protocol.json",
        {
            "seed": "full_train" if args.full else args.seed,
            "encoder": meta["model"],
            "revision": meta["revision"],
            "sample_per_class": 128,
            "training_posts": 384,
            "training_pairs": 768,
            "pair_batch_size": 16,
            "epochs": 1,
            "steps": 48,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "max_tokens": 128,
            "random_seed": 1234,
            "loss": "MSE(cosine_similarity(left,right), same_class_target)",
            "downstream": "same balanced HGB with history and BM25; also fixed 50/50 mean with established SVD hybrid",
            "gate": "Continue to seeds 7 and 2026 only if a new candidate exceeds the primary incumbent ensemble by at least 0.002 macro-F1.",
            "validation_labels_used_for_encoder_training": False,
            "official_test_not_used_in_pilot": not args.full,
            "official_test_labels_used_for_encoder_training": False,
            "selection_is_adaptive": True,
        },
    )
    print(
        "Prepared 768 balanced same/different-class pairs from 384 fit-only posts.",
        flush=True,
    )
else:
    metadata = json.loads((root / "encoder_training.json").read_text())
    assert (
        metadata["input_sha256"]
        == hashlib.sha256(
            Path("results/fusion/encoder_input.jsonl").read_bytes()
        ).hexdigest()
    )
    vectors = np.load(root / "embeddings.npy")
    assert vectors.shape == (len(table), 384)
    table["age_days"] = table.raw_age.fillna(table.loc[fit, "raw_age"].median())
    with threadpool_limits(limits=4):
        adapted = probabilities(table, vectors, config, fit, valid, "balanced")
    oldroot = Path("results/round3")
    saved = pd.read_csv(oldroot / f"seed{args.seed}_svd_balanced_predictions.csv")
    assert np.array_equal(
        saved[["author", "post"]].to_numpy(),
        table.loc[valid, ["author", "post"]].to_numpy(),
    )
    svd = np.load(oldroot / f"seed{args.seed}_svd_balanced_probabilities.npy")
    incumbent = np.load(
        oldroot / f"seed{args.seed}_equal_weight_ensemble_probabilities.npy"
    )
    control = np.load(
        oldroot / f"seed{args.seed}_embeddings_balanced_probabilities.npy"
    )
    rows = []
    for name, p in [
        ("frozen_embedding_control", control),
        ("incumbent_ensemble", incumbent),
        ("adapted_embedding_hybrid", adapted),
        ("adapted_equal_weight_ensemble", (adapted + svd) / 2),
    ]:
        assert np.allclose(p.sum(axis=1), 1) and np.isfinite(p).all()
        pred = p.argmax(axis=1)
        row = {
            "experiment": name,
            **score(table.loc[valid, "label"], pred),
            "macro_f1_unrounded": float(
                f1_score(table.loc[valid, "label"], pred, average="macro")
            ),
        }
        rows.append(row)
        np.save(root / f"{name}_probabilities.npy", p)
        print(row, flush=True)
    best = max(rows[2:], key=lambda r: r["macro_f1_unrounded"])
    gain = best["macro_f1_unrounded"] - rows[1]["macro_f1_unrounded"]
    write_json(
        root / "assessment.json",
        {
            "seed": args.seed,
            "results": rows,
            "best_new_candidate": best["experiment"],
            "gain_over_incumbent": gain,
            "primary_gate_applicable": args.seed == 42,
            "gate_passed": gain >= 0.002 if args.seed == 42 else None,
            "conclusion": (
                (
                    "Proceed to additional author splits"
                    if gain >= 0.002
                    else "Stop bounded pilot; preserve the already validated final ensemble"
                )
                if args.seed == 42
                else "Confirmation split recorded; final decision uses the three-split mean"
            ),
            "not_a_blind_test": True,
        },
    )
    print(
        "Primary gate passed:" if args.seed == 42 else "Confirmation split complete:",
        gain >= 0.002 if args.seed == 42 else args.seed,
        "; gain over incumbent:",
        gain,
        flush=True,
    )
