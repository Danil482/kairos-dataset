"""Bounded final comparison: two strong models, their mean, and natural weights.

Run validation first. The winning configuration is frozen before test loading.
No threshold search, blending-weight search, or encoder fine-tuning is performed.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize
from threadpoolctl import threadpool_limits

from baseline import model
from kairos.metrics import score
from kairos.recent_features import RECENT_COLUMNS

ROOT = Path("results/fusion")
OUT = Path("results/round3")
CANDIDATES = [
    "embeddings_balanced",
    "svd_balanced",
    "equal_weight_ensemble",
    "embeddings_natural",
]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def training_data():
    posts = pd.read_pickle(ROOT / "train_posts.pkl")
    if not (ROOT / "train_recent_table.pkl").exists():
        from kairos.recent_features import recent_author_features

        base = pd.read_pickle(ROOT / "train_table.pkl")
        table = base.merge(
            recent_author_features(posts),
            on=["author", "post"],
            how="left",
            validate="one_to_one",
        )
        assert table[RECENT_COLUMNS].notna().all().all()
        table.to_pickle(ROOT / "train_recent_table.pkl")
    else:
        table = pd.read_pickle(ROOT / "train_recent_table.pkl")
    raw = posts[["author", "post"]].assign(
        raw_age=(posts.crawled_at - posts.posted_at).dt.total_seconds() / 86400
    )
    before = table[["author", "post"]].copy()
    table = table.merge(raw, on=["author", "post"], validate="one_to_one")
    assert table[["author", "post"]].equals(before)
    assert table.index.equals(pd.RangeIndex(len(table)))
    config = json.loads((ROOT / "feature_config.json").read_text())
    meta = json.loads((ROOT / "embedding_metadata.json").read_text())
    assert (
        meta["input_sha256"]
        == hashlib.sha256((ROOT / "encoder_input.jsonl").read_bytes()).hexdigest()
    )
    keys = pd.read_json(ROOT / "encoder_input.jsonl", lines=True)[["author", "post"]]
    assert keys.equals(before), "Embedding row order differs from training table"
    vectors = np.load(ROOT / "multilingual_embeddings.npy")
    assert vectors.shape == (len(table), 384) and np.isfinite(vectors).all()
    return table, vectors, config


def frame_with_vectors(table, vectors):
    names = [f"text_dimension_{i}" for i in range(vectors.shape[1])]
    return (
        pd.concat(
            [table.reset_index(drop=True), pd.DataFrame(vectors, columns=names)], axis=1
        ),
        names,
    )


def classifier(config, names, class_weight):
    numeric = config["numeric"] + RECENT_COLUMNS + names
    categorical = config["categorical"]
    candidate = model()
    candidate.set_params(
        prep=ColumnTransformer(
            [
                ("num", StandardScaler(), numeric),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", min_frequency=20),
                    categorical,
                ),
            ]
        ),
        clf__class_weight=class_weight,
    )
    return candidate, numeric + categorical


def probabilities(table, vectors, config, fit, valid, class_weight):
    frame, names = frame_with_vectors(table, vectors)
    candidate, columns = classifier(config, names, class_weight)
    candidate.fit(frame.loc[fit, columns], frame.loc[fit, "label"])
    assert np.array_equal(candidate.classes_, [0, 1, 2])
    return candidate.predict_proba(frame.loc[valid, columns])


def svd_vectors(table, fit, valid):
    vectorizer = TfidfVectorizer(min_df=5, max_features=20000, dtype=np.float32)
    xfit = vectorizer.fit_transform(table.loc[fit, "text"])
    reducer = TruncatedSVD(n_components=128, n_iter=7, random_state=42)
    vectors = np.empty((len(table), 128), dtype=np.float32)
    vectors[fit] = normalize(reducer.fit_transform(xfit))
    vectors[valid] = normalize(
        reducer.transform(vectorizer.transform(table.loc[valid, "text"]))
    )
    return vectors


def cached_primary(table, valid, old_name):
    saved = pd.read_csv(ROOT / f"{old_name}_predictions.csv")
    assert np.array_equal(
        saved[["author", "post"]].to_numpy(),
        table.loc[valid, ["author", "post"]].to_numpy(),
    )
    p = np.load(ROOT / f"{old_name}_probabilities.npy")
    assert p.shape == (int(valid.sum()), 3) and np.isfinite(p).all()
    assert np.array_equal(saved.label, p.argmax(axis=1))
    assert np.allclose(p.sum(axis=1), 1)
    return p


def validate(reuse_primary_cache=True):
    OUT.mkdir(parents=True, exist_ok=True)
    table, embeddings, config = training_data()
    write_json(
        OUT / "protocol.json",
        {
            "candidates": CANDIDATES,
            "seeds": [42, 7, 2026],
            "selection": "highest mean unrounded macro-F1 across the three author splits",
            "ensemble_weights": [0.5, 0.5],
            "extra_experiment": "same embedding hybrid with class_weight=None; all other settings fixed",
            "test_seen_in_earlier_rounds": True,
            "primary_equal_weight_result_seen_before_this_protocol": 0.6096,
            "no_claim_of_independent_validation": "These reused overlapping author splits are for model selection, not an unbiased estimate.",
        },
    )
    results = []
    for seed in [42, 7, 2026]:
        _, valid_authors = train_test_split(
            sorted(table.author.unique()), test_size=0.2, random_state=seed
        )
        valid = table.author.isin(valid_authors).to_numpy()
        fit = ~valid
        assert set(table.loc[fit, "author"]).isdisjoint(table.loc[valid, "author"])
        table["age_days"] = table.raw_age.fillna(table.loc[fit, "raw_age"].median())
        cached_names = ["bm25_recent_embeddings384", "bm25_recent_svd128"]
        reuse_primary = (
            reuse_primary_cache
            and seed == 42
            and all(
                (ROOT / f"{n}_{suffix}").exists()
                for n in cached_names
                for suffix in ["probabilities.npy", "predictions.csv"]
            )
        )
        if seed == 42:
            assert np.array_equal(valid, table.validation.to_numpy())
        if reuse_primary:
            emb = cached_primary(table, valid, "bm25_recent_embeddings384")
            svd = cached_primary(table, valid, "bm25_recent_svd128")
        else:
            print(f"Seed {seed}: fitting the two established hybrid models", flush=True)
            emb = probabilities(table, embeddings, config, fit, valid, "balanced")
            svd = probabilities(
                table, svd_vectors(table, fit, valid), config, fit, valid, "balanced"
            )
        print(f"Seed {seed}: testing natural class frequencies", flush=True)
        natural = probabilities(table, embeddings, config, fit, valid, None)
        all_probs = dict(zip(CANDIDATES, [emb, svd, (emb + svd) / 2, natural]))
        for name, p in all_probs.items():
            assert p.shape == (int(valid.sum()), 3) and np.isfinite(p).all()
            assert np.allclose(p.sum(axis=1), 1)
            predicted = p.argmax(axis=1)
            row = {
                "seed": seed,
                "experiment": name,
                "n_fit": int(fit.sum()),
                "n_valid": int(valid.sum()),
                **score(table.loc[valid, "label"], predicted),
                "macro_f1_unrounded": float(
                    f1_score(table.loc[valid, "label"], predicted, average="macro")
                ),
            }
            results.append(row)
            np.save(OUT / f"seed{seed}_{name}_probabilities.npy", p)
            table.loc[valid, ["author", "post", "label"]].rename(
                columns={"label": "true_label"}
            ).assign(predicted_label=predicted).to_csv(
                OUT / f"seed{seed}_{name}_predictions.csv", index=False
            )
            print(row, flush=True)
        write_json(OUT / "validation_results.json", results)
    summary = []
    for name in CANDIDATES:
        rows = [r for r in results if r["experiment"] == name]
        summary.append(
            {
                "experiment": name,
                "macro_f1_by_seed_42_7_2026": [r["macro_f1_unrounded"] for r in rows],
                "mean_macro_f1": float(
                    np.mean([r["macro_f1_unrounded"] for r in rows])
                ),
            }
        )
    selected = max(summary, key=lambda r: r["mean_macro_f1"])["experiment"]
    write_json(
        OUT / "selection.json",
        {
            "selected": selected,
            "candidates": summary,
            "selection_rule": "highest mean unrounded macro-F1 across seeds 42,7,2026",
            "official_test_used_for_round3_selection": False,
            "test_seen_in_earlier_rounds": True,
        },
    )
    print("Frozen selection:", selected, summary, flush=True)


def final():
    # Selection must exist before the original test data are read.
    selection = json.loads((OUT / "selection.json").read_text())
    selected = selection["selected"]
    assert selected in CANDIDATES
    from kairos.data import load_labels, load_posts, load_followers
    from kairos.label import author_level
    from kairos import features
    from kairos.bm25_features import author_bm25_features
    from kairos.recent_features import recent_author_features

    table, embeddings, config = training_data()
    age_fill = float(table.raw_age.median())
    table["age_days"] = table.raw_age.fillna(age_fill)
    data = Path(os.environ.get("KAIROS_DATA_DIR", "data"))
    truth = load_labels(data).query("split == 'test'").copy()
    assert set(table.author).isdisjoint(truth.author)
    posts = load_posts(data)
    posts = posts[posts.author.isin(truth.author)].copy()
    test = features.build(author_level(posts), load_followers(data))
    text = test.text.fillna("")
    letters = text.map(lambda s: sum(c.isalpha() for c in s))
    upper = text.map(lambda s: sum(c.isupper() for c in s))
    words = text.str.findall(r"[^\W\d_]+")
    test["n_exclamations"] = text.str.count("!")
    test["uppercase_ratio"] = upper / letters.clip(lower=1)
    test["avg_word_length"] = words.map(lambda v: sum(map(len, v)) / max(1, len(v)))
    test["age_days"] = (
        (test.crawled_at - test.posted_at).dt.total_seconds() / 86400
    ).fillna(age_fill)
    test = test.merge(
        author_bm25_features(posts)[
            ["author", "post", "bm25_history_delta", "bm25_match_count"]
        ],
        on=["author", "post"],
        validate="one_to_one",
    )
    test = test.merge(
        recent_author_features(posts), on=["author", "post"], validate="one_to_one"
    )
    test = truth.merge(test, on=["author", "post"], validate="one_to_one").reset_index(
        drop=True
    )
    # Reuse frozen test embeddings only if their full input identity still matches.
    test.to_pickle(OUT / "test_table.pkl")
    if selected != "svd_balanced":
        test_input = test[["author", "post", "text"]]
        test_input.to_json(
            ROOT / "test_encoder_input.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
        )
        encoder_python = Path(".venv-embeddings/bin/python")
        subprocess.run(
            [
                (
                    str(encoder_python.absolute())
                    if encoder_python.exists()
                    else sys.executable
                ),
                "encode_posts.py",
                "--input",
                str(ROOT / "test_encoder_input.jsonl"),
                "--prefix",
                "test_",
            ],
            check=True,
        )
        cached_input = pd.read_json(ROOT / "test_encoder_input.jsonl", lines=True)
        assert test_input.equals(
            cached_input
        ), "Cached test embeddings do not match current posts"
        meta = json.loads((ROOT / "test_embedding_metadata.json").read_text())
        assert (
            meta["input_sha256"]
            == hashlib.sha256(
                (ROOT / "test_encoder_input.jsonl").read_bytes()
            ).hexdigest()
        )
        assert (
            meta["revision"]
            == json.loads((ROOT / "embedding_metadata.json").read_text())["revision"]
        )
        test_embeddings = np.load(ROOT / "test_multilingual_embeddings.npy")
        assert test_embeddings.shape == (len(test), 384)
    models_to_fit = (
        ["embeddings", "svd"]
        if selected == "equal_weight_ensemble"
        else ["svd" if selected == "svd_balanced" else "embeddings"]
    )
    probability_list = []
    for representation in models_to_fit:
        if representation == "svd":
            vectorizer = TfidfVectorizer(min_df=5, max_features=20000, dtype=np.float32)
            reducer = TruncatedSVD(n_components=128, n_iter=7, random_state=42)
            train_vectors = normalize(
                reducer.fit_transform(vectorizer.fit_transform(table.text))
            )
            test_vectors = normalize(reducer.transform(vectorizer.transform(test.text)))
        else:
            train_vectors, test_vectors = embeddings, test_embeddings
        train_frame, names = frame_with_vectors(table, train_vectors)
        test_frame, test_names = frame_with_vectors(test, test_vectors)
        assert names == test_names
        candidate, columns = classifier(
            config, names, None if selected == "embeddings_natural" else "balanced"
        )
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
        print("Full-train fit:", representation, len(train_frame), flush=True)
        candidate.fit(train_frame[columns], train_frame.label)
        assert np.array_equal(candidate.classes_, [0, 1, 2])
        probability_list.append(candidate.predict_proba(test_frame[columns]))
    p = np.mean(probability_list, axis=0)
    predicted = p.argmax(axis=1)
    submission = test[["author", "post"]].assign(label=predicted.astype(int))
    assert (
        len(submission) == len(truth)
        and not submission.duplicated(["author", "post"]).any()
    )
    assert set(map(tuple, submission[["author", "post"]].to_numpy())) == set(
        map(tuple, truth[["author", "post"]].to_numpy())
    )
    Path("results/predictions").mkdir(parents=True, exist_ok=True)
    submission.to_csv("results/predictions/lab1_final_v3.csv", index=False)
    np.save(OUT / "test_probabilities.npy", p)
    pr, recall, f1, support = precision_recall_fscore_support(
        test.label, predicted, labels=[0, 1, 2], zero_division=0
    )
    report = {
        "selected": selected,
        "n_train": len(table),
        "n_test": len(test),
        "age_fill": age_fill,
        "results": score(test.label, predicted),
        "confusion_rows_true": confusion_matrix(
            test.label, predicted, labels=[0, 1, 2]
        ).tolist(),
        "per_class": [
            {
                "class": name,
                "precision": float(pr[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(["below", "typical", "above"])
        ],
        "selection": selection,
        "caveat": "Official test was seen in previous rounds. Round-3 configuration was frozen by validation before this follow-up assessment. No blind-test claim.",
    }
    write_json(OUT / "test_results.json", report)
    print("Follow-up official test:", report["results"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["validation", "final"], required=True)
    parser.add_argument(
        "--rebuild-primary",
        action="store_true",
        help="Refit seed 42 instead of reusing the verified earlier probability files.",
    )
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        if args.stage == "validation":
            validate(reuse_primary_cache=not args.rebuild_primary)
        else:
            final()
