"""Freeze the adapted-encoder selection, then assess only the chosen final model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from kairos.metrics import score
from lab_round3 import training_data, classifier, frame_with_vectors, write_json

ROOT = Path("results/adaptation")
parser = argparse.ArgumentParser()
parser.add_argument("--stage", choices=["select", "final"], required=True)
args = parser.parse_args()

if args.stage == "select":
    names = [
        "incumbent_ensemble",
        "adapted_embedding_hybrid",
        "adapted_equal_weight_ensemble",
    ]
    checks = [
        json.loads((ROOT / f"seed{seed}/assessment.json").read_text())
        for seed in [42, 7, 2026]
    ]
    summary = []
    for name in names:
        values = [
            next(
                r["macro_f1_unrounded"]
                for r in check["results"]
                if r["experiment"] == name
            )
            for check in checks
        ]
        summary.append(
            {
                "experiment": name,
                "macro_f1_by_seed_42_7_2026": values,
                "mean_macro_f1": float(np.mean(values)),
            }
        )
    chosen = max(summary, key=lambda r: r["mean_macro_f1"])["experiment"]
    selection = {
        "selected": chosen,
        "candidates": summary,
        "selection_rule": "highest mean unrounded macro-F1 across author splits 42,7,2026",
        "test_seen_in_previous_rounds": True,
        "official_test_used_for_this_selection": False,
        "encoder_configuration_fixed_across_splits": True,
        "split_specific_encoder_training": True,
    }
    write_json(ROOT / "selection.json", selection)
    print(json.dumps(selection, indent=2), flush=True)
else:
    selection = json.loads((ROOT / "selection.json").read_text())
    chosen = selection["selected"]
    if chosen == "incumbent_ensemble":
        print(
            "The incumbent remains selected; no new official-test evaluation.",
            flush=True,
        )
        raise SystemExit(0)
    table, _, config = training_data()
    table["age_days"] = table.raw_age.fillna(table.raw_age.median())
    test = pd.read_pickle("results/round3/test_table.pkl").reset_index(drop=True)
    assert set(table.author).isdisjoint(test.author)
    full = ROOT / "full"
    metadata = json.loads((full / "encoder_training.json").read_text())
    train_input = Path("results/fusion/encoder_input.jsonl")
    test_input = Path("results/fusion/test_encoder_input.jsonl")
    assert (
        metadata["input_sha256"] == hashlib.sha256(train_input.read_bytes()).hexdigest()
    )
    assert (
        metadata["test_input_sha256"]
        == hashlib.sha256(test_input.read_bytes()).hexdigest()
    )
    assert test[["author", "post", "text"]].equals(pd.read_json(test_input, lines=True))
    sample = pd.read_csv(full / "training_sample.csv")
    assert set(sample.author).isdisjoint(test.author)
    vectors = np.load(full / "embeddings.npy")
    test_vectors = np.load(full / "test_embeddings.npy")
    assert vectors.shape == (len(table), 384) and test_vectors.shape == (len(test), 384)
    with threadpool_limits(limits=4):
        train_frame, names = frame_with_vectors(table, vectors)
        test_frame, _ = frame_with_vectors(test, test_vectors)
        candidate, columns = classifier(config, names, "balanced")
        assert not set(
            [
                "label",
                "reactions",
                "resid",
                "log_reactions",
                "author",
                "post",
                "raw_age",
            ]
        ) & set(columns)
        print("Fitting full-train classifier on adapted text + history...", flush=True)
        candidate.fit(train_frame[columns], train_frame.label)
        assert np.array_equal(candidate.classes_, [0, 1, 2])
        p = candidate.predict_proba(test_frame[columns])
        if chosen == "adapted_equal_weight_ensemble":
            print("Refitting the fixed SVD companion on full train...", flush=True)
            vectorizer = TfidfVectorizer(min_df=5, max_features=20000, dtype=np.float32)
            reducer = TruncatedSVD(n_components=128, n_iter=7, random_state=42)
            train_svd = normalize(
                reducer.fit_transform(vectorizer.fit_transform(table.text))
            )
            test_svd = normalize(reducer.transform(vectorizer.transform(test.text)))
            train_frame, names = frame_with_vectors(table, train_svd)
            test_frame, _ = frame_with_vectors(test, test_svd)
            companion, columns = classifier(config, names, "balanced")
            companion.fit(train_frame[columns], train_frame.label)
            assert np.array_equal(companion.classes_, [0, 1, 2])
            p = (p + companion.predict_proba(test_frame[columns])) / 2
    assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1)
    pred = p.argmax(axis=1)
    submission = test[["author", "post"]].assign(label=pred.astype(int))
    assert (
        len(submission) == 17623 and not submission.duplicated(["author", "post"]).any()
    )
    submission.to_csv("results/predictions/lab1_final_v4.csv", index=False)
    np.save(ROOT / "test_probabilities.npy", p)
    pr, re, f1, support = precision_recall_fscore_support(
        test.label, pred, labels=[0, 1, 2], zero_division=0
    )
    result = {
        "selected": chosen,
        "n_train": len(table),
        "n_test": len(test),
        "results": score(test.label, pred),
        "selection": selection,
        "confusion_rows_true": confusion_matrix(
            test.label, pred, labels=[0, 1, 2]
        ).tolist(),
        "per_class": [
            {
                "class": name,
                "precision": float(pr[i]),
                "recall": float(re[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(["below", "typical", "above"])
        ],
        "encoder_training_posts": 384,
        "full_train_classifier_posts": len(table),
        "caveat": "Official test was seen in prior rounds. Adaptation selection was fixed by reused validation splits before this follow-up evaluation; no blind-test claim.",
    }
    write_json(ROOT / "test_results.json", result)
    print("Adapted follow-up test:", result["results"], flush=True)
