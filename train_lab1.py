from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".cache" / "matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) // 2)))

import joblib
import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

import baseline
from kairos.data import load_followers, load_labels, load_posts
from kairos.label import author_level
from kairos.metrics import score
from lab1_features import CATEGORICAL, build
from lab1_metadata import enrich_posts
from lab1_uncertainty import author_bootstrap
from lab1_embeddings import CONTEXT_NUM, EmbeddingProjection, EmbeddingStore

SEED = 42
RELEASE_COUNTS = {"train_posts": 71395, "test_posts": 17623,
                  "train_authors": 3295, "test_authors": 824}


def text_model(numeric, c=1.0):
    return Pipeline([
        ("features", ColumnTransformer([
            ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=.98,
                                     max_features=40000, sublinear_tf=True,
                                     strip_accents="unicode", dtype=np.float32), "clean_text"),
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                     min_df=5, max_features=50000,
                                     sublinear_tf=True, dtype=np.float32), "clean_text"),
            ("profile", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=15000,
                                       sublinear_tf=True, dtype=np.float32), "profile_text"),
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                                  ("scale", StandardScaler())]), numeric),
            ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ], sparse_threshold=1.0)),
        ("classifier", LogisticRegression(C=c, class_weight="balanced", max_iter=1000,
                                           solver="lbfgs", random_state=SEED)),
    ])


def boosted(iterations, threads, depth=6):
    return CatBoostClassifier(iterations=iterations, depth=depth, learning_rate=.05,
                              l2_leaf_reg=5, loss_function="MultiClass",
                              auto_class_weights="Balanced", random_seed=SEED,
                              thread_count=threads, allow_writing_files=False, verbose=False)


def tabular(table, numeric):
    out = table[numeric + CATEGORICAL].copy()
    out[numeric] = out[numeric].fillna(-999.0)
    return out


def text_fingerprint(table, numeric):
    columns = numeric + CATEGORICAL + ["author", "post", "clean_text", "profile_text", "label"]
    values = pd.util.hash_pandas_object(table[columns], index=False).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def baseline_table(data):
    levelled = author_level(load_posts(data))
    values = baseline.features.build(levelled, load_followers(data))
    return load_labels(data).merge(values, on=["author", "post"], how="left", validate="one_to_one")


def select_blend(y, tree_prob, text_prob):
    best = None
    y = np.asarray(y, dtype=int)
    class_grid = [.45, .6, .75, .9, 1.05, 1.2]

    def evaluate(tree_weight, below_weight, above_weight):
        nonlocal best
        tree_weight = np.asarray(tree_weight, dtype=float)
        weights = np.array([below_weight, 1., above_weight])
        prob = tree_weight * tree_prob + (1 - tree_weight) * text_prob
        pred = np.argmax(prob * weights, axis=1)
        matrix = np.bincount(y * 3 + pred, minlength=9).reshape(3, 3)
        denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
        metric = np.divide(2 * matrix.diagonal(), denominator,
                           out=np.zeros(3), where=denominator > 0).mean()
        if best is None or metric > best[0]:
            w = tree_weight.tolist()
            best = (float(metric), w, weights.tolist())

    for values in itertools.product(np.linspace(0, 1, 11), class_grid, class_grid):
        evaluate(*values)
    centre = float(best[1])
    nearby = sorted(set(round(float(np.clip(centre + delta, 0, 1)), 2) for delta in [-.2, 0, .2]))
    for w0, w1, w2, below, above in itertools.product(nearby, nearby, nearby, class_grid, class_grid):
        evaluate([w0, w1, w2], below, above)
    return {"validation_macro_f1": best[0], "tree_weight": best[1], "class_weights": best[2]}


def blend(tree_prob, text_prob, config):
    w = np.asarray(config["tree_weight"], dtype=float)
    return np.argmax((w * tree_prob + (1 - w) * text_prob) * config["class_weights"], axis=1)


def checked_table(data):
    data = Path(data)
    source = [Path(__file__), Path(__file__).with_name("lab1_features.py"),
              Path(__file__).with_name("lab1_metadata.py"),
              Path(__file__).parent / "kairos" / "features.py",
              Path(__file__).parent / "kairos" / "label.py",
              Path(__file__).parent / "kairos" / "data.py"]
    fingerprint = hashlib.sha256()
    for path in source:
        fingerprint.update(path.read_bytes())
    for path in [data / "labels.csv", data / "split.json"]:
        fingerprint.update(path.read_bytes())
    for directory in ["posts", "authors"]:
        for path in sorted((data / directory).glob("*.json")):
            stat = path.stat()
            fingerprint.update(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    manifest = data / "media" / "manifest.jsonl"
    if manifest.is_file():
        stat = manifest.stat()
        fingerprint.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    cache = Path(__file__).parent / ".cache" / f"table-{fingerprint.hexdigest()[:20]}.joblib"
    if cache.is_file():
        return joblib.load(cache)
    labels = load_labels(data)
    if labels.duplicated(["author", "post"]).any():
        raise ValueError("Duplicate release labels")
    if not labels["label"].isin([0, 1, 2]).all():
        raise ValueError("Invalid release class")
    if set(labels.loc[labels.split == "train", "author"]) & set(labels.loc[labels.split == "test", "author"]):
        raise ValueError("Official split has overlapping authors")
    raw = enrich_posts(load_posts(data), data)
    values, numeric = build(author_level(raw), load_followers(data))
    table = labels.merge(values, on=["author", "post"], how="left", validate="one_to_one")
    if table["level"].isna().any():
        raise ValueError("Incomplete release: some labelled posts lack earlier history")
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_name(cache.name + f".{os.getpid()}.tmp")
    joblib.dump((table, numeric), temporary, compress=3)
    temporary.replace(cache)
    return table, numeric


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--output", type=Path, default=Path("results/lab1"))
    ap.add_argument("--iterations", type=int, default=1200)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--text-c", type=float, default=1.0)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--embeddings", type=Path)
    ap.add_argument("--post-components", type=int, default=64)
    ap.add_argument("--profile-components", type=int, default=16)
    ap.add_argument("--reuse-text-validation", type=Path)
    ap.add_argument("--selection", type=Path)
    ap.add_argument("--allow-fixture", action="store_true",
                    help="allow a small synthetic integration fixture, never for submission")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    table, numeric = checked_table(args.data)
    text_numeric = numeric.copy()
    store = EmbeddingStore(args.embeddings) if args.embeddings else None
    if store:
        table = store.context(table)
        numeric = numeric + CONTEXT_NUM
    train = table[table.split == "train"].reset_index(drop=True)
    test = table[table.split == "test"].reset_index(drop=True)
    text_input_hash = text_fingerprint(train, text_numeric)
    counts = {"train_posts": len(train), "test_posts": len(test),
              "train_authors": int(train.author.nunique()), "test_authors": int(test.author.nunique())}
    official = counts == RELEASE_COUNTS
    if not official and not args.allow_fixture:
        raise SystemExit(f"Release counts differ from official data: {counts}. "
                         "Check extraction; --allow-fixture is only for synthetic tests.")
    fit_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED)
                            .split(train, groups=train.author))
    fit, val = train.iloc[fit_idx], train.iloc[val_idx]
    reference = baseline_table(args.data)
    reference_train = reference[reference.split == "train"].reset_index(drop=True)
    reference_test = reference[reference.split == "test"].reset_index(drop=True)
    if set(fit.author) & set(val.author):
        raise AssertionError("Validation author leakage")
    print(f"fit={len(fit)}, validation={len(val)}, official test={len(test)}", flush=True)
    validation = {}
    projection = None
    if store:
        projection = EmbeddingProjection(args.post_components, args.profile_components)
        if not args.selection:
            projection.fit(fit, store)
            fit, extra_numeric = projection.transform(fit, store)
            val, _ = projection.transform(val, store)
            numeric = numeric + extra_numeric
    with threadpool_limits(limits=args.threads):
        if args.selection:
            selected = json.loads(args.selection.read_text())
            expected = numeric.copy()
            if store:
                expected += [f"e5_post_pc{i:03d}" for i in range(args.post_components)]
                expected += [f"e5_profile_pc{i:03d}" for i in range(args.profile_components)]
            if (selected["numeric_features"] != expected
                    or selected["text_numeric_features"] != text_numeric
                    or selected["text_input_sha256"] != text_input_hash
                    or selected["catboost_depth"] != args.depth
                    or selected["text_c"] != args.text_c
                    or selected["validation_authors"] != sorted(val.author.unique())):
                raise ValueError("Frozen selection does not match this data, split or feature configuration")
            if store and any(selected["embeddings"].get(k) != store.info.get(k)
                             for k in ["revision", "input_sha256", "history_input_sha256"]):
                raise ValueError("Frozen selection has different encoder inputs")
            numeric = expected
            iterations = selected["catboost_iterations"]
            config = selected["blend"]
            validation = selected["validation"]
            (args.output / "selection.json").write_text(json.dumps(selected, indent=2))
            print("Loaded frozen selection; no validation settings will be retuned.", flush=True)
        else:
            ref = baseline.model()
            ref.fit(reference_train.iloc[fit_idx], fit.label)
            validation["official_baseline"] = score(val.label, ref.predict(reference_train.iloc[val_idx]))
            print("Baseline validation:", validation["official_baseline"], flush=True)

            tree = boosted(args.iterations, args.threads, args.depth)
            tree.fit(tabular(fit, numeric), fit.label, cat_features=CATEGORICAL,
                     eval_set=(tabular(val, numeric), val.label),
                     early_stopping_rounds=100, use_best_model=True, verbose=200)
            iterations = max(1, tree.tree_count_)
            tree_prob = tree.predict_proba(tabular(val, numeric))
            validation["history_catboost"] = score(val.label, tree_prob.argmax(axis=1))

            linear = None
            if args.reuse_text_validation:
                previous = np.load(args.reuse_text_validation / "validation_probabilities.npz")
                prior_config = json.loads((args.reuse_text_validation / "selection.json").read_text())
                if (not np.array_equal(previous["authors"], val.author)
                        or not np.array_equal(previous["labels"], val.label)
                        or "posts" not in previous.files
                        or not np.array_equal(previous["posts"], val.post)
                        or prior_config.get("text_c", 1.0) != args.text_c
                        or prior_config.get("text_input_sha256") != text_input_hash
                        or prior_config.get("text_numeric_features", prior_config["numeric_features"]) != text_numeric):
                    raise ValueError("Saved text probabilities have a different split or feature configuration")
                text_prob = previous["text"]
            else:
                linear = text_model(text_numeric, args.text_c)
                linear.fit(fit, fit.label)
                text_prob = linear.predict_proba(val)
            validation["text_and_metadata"] = score(val.label, text_prob.argmax(axis=1))
            config = select_blend(val.label, tree_prob, text_prob)
            validation["selected_blend"] = score(val.label, blend(tree_prob, text_prob, config))
            print("Validation results:", json.dumps(validation), flush=True)

            selected = {"seed": SEED, "catboost_iterations": iterations, "blend": config,
                        "catboost_depth": args.depth, "text_c": args.text_c,
                        "embeddings": store.info if store else None,
                        "post_components": args.post_components if store else None,
                        "profile_components": args.profile_components if store else None,
                        "numeric_features": numeric, "categorical_features": CATEGORICAL,
                        "text_numeric_features": text_numeric,
                        "text_input_sha256": text_input_hash,
                        "validation": validation,
                        "fit_authors": sorted(fit.author.unique()),
                        "validation_authors": sorted(val.author.unique())}
            (args.output / "selection.json").write_text(json.dumps(selected, indent=2))
            np.savez_compressed(args.output / "validation_probabilities.npz", tree=tree_prob,
                                text=text_prob, labels=val.label.to_numpy(), authors=val.author.to_numpy(dtype=str),
                                posts=val.post.to_numpy(dtype=str))
            del ref, tree, linear
        if args.validate_only:
            print("Validation only: official test labels were not scored.", flush=True)
            return

        print("Refitting frozen models on all train authors...", flush=True)
        if store:
            projection.fit(train, store)
            train, _ = projection.transform(train, store)
            test, _ = projection.transform(test, store)
        ref = baseline.model().fit(reference_train, train.label)
        baseline_pred = ref.predict(reference_test)
        del ref
        tree = boosted(iterations, args.threads, args.depth)
        tree.fit(tabular(train, numeric), train.label, cat_features=CATEGORICAL, verbose=200)
        tree_test = tree.predict_proba(tabular(test, numeric))
        linear = text_model(text_numeric, args.text_c).fit(train, train.label)
        text_test = linear.predict_proba(test)
        predictions = {
            "official_baseline": baseline_pred,
            "history_catboost": tree_test.argmax(axis=1),
            "text_and_metadata": text_test.argmax(axis=1),
            "selected_blend": blend(tree_test, text_test, config),
        }

        results = {}
        for name, pred in predictions.items():
            results[name] = {**score(test.label, pred),
                             "confusion_matrix": confusion_matrix(test.label, pred, labels=[0, 1, 2]).tolist(),
                             "classification_report": classification_report(
                                 test.label, pred, labels=[0, 1, 2],
                                 target_names=["below", "typical", "above"],
                                 output_dict=True, zero_division=0)}
            test[["author", "post"]].assign(label=pred).to_csv(args.output / f"{name}.csv", index=False)
        joblib.dump({"tree": tree, "text": linear, "config": config,
                     "numeric": numeric, "embedding_projection": projection,
                     "embedding_info": store.info if store else None},
                    args.output / "model.joblib", compress=3)
        pd.Series(tree.feature_importances_, index=numeric + CATEGORICAL).sort_values(
            ascending=False).to_csv(args.output / "feature_importance.csv", header=["importance"])
    results["always_typical"] = score(test.label, np.ones(len(test), dtype=int))
    manifest = {
        "author": "Rostislav Manzyuk", "github": "grinchaus", "seed": SEED,
        "data_kind": "official-release" if official else "synthetic-fixture",
        "train_posts": len(train), "test_posts": len(test),
        "train_authors": int(train.author.nunique()), "test_authors": int(test.author.nunique()),
        "validation_posts": len(val), "validation_authors": int(val.author.nunique()),
        "elapsed_seconds": round(time.time() - started, 1),
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "labels_sha256": hashlib.sha256((args.data / "labels.csv").read_bytes()).hexdigest(),
        "data_quality": {"labelled_missing_post_dates": int(table.posted_at.isna().sum()),
                         "labelled_missing_crawl_dates": int(table.crawled_at.isna().sum()),
                         "crawl_timestamp_parser": "mixed ISO-8601, UTC; original baseline unchanged"},
        "selected_config": config, "catboost_iterations": iterations,
        "catboost_depth": args.depth, "text_c": args.text_c,
        "requested_iterations": args.iterations,
        "embeddings": store.info if store else None,
        "post_components": args.post_components if store else None,
        "profile_components": args.profile_components if store else None,
        "confidence_intervals": author_bootstrap(test.author, test.label, predictions),
        "validation": validation, "test": results,
        "limitations": ["Reaction history is a crawl-time snapshot, as in the official baseline.",
                        "Current follower counts and crawl age are collection artefacts.",
                        "Images and video are not used.",
                        "Validation model selection scores are optimistic after the small blend search."]}
    (args.output / "metrics.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: {m: v for m, v in x.items() if m in ["macro_f1", "qwk", "accuracy"]}
                      for k, x in results.items()}, indent=2), flush=True)
    print("Saved predictions, metrics and model to", args.output, flush=True)


if __name__ == "__main__":
    main()
