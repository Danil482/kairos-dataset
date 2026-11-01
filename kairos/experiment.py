"""Reproduce author-held-out experiments, refit, and write official predictions.

    python -m kairos.experiment --config configs/cpu.json
    python evaluate.py --predictions results/cpu/final/predictions.csv

Test labels are checked for release integrity, never used for model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
import time
import zipfile
import zlib
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from kairos.experiment_features import load_experiment_data
from kairos.experiment_uncertainty import paired_author_bootstrap
from kairos.experiment_text import (fit_svd, history_feature_columns,
                                   history_vector_features, make_text_model)

ROOT = Path(__file__).resolve().parent.parent
KEYS = ["author", "post"]


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def metrics(truth, predicted) -> dict:
    return {
        "macro_f1": float(f1_score(truth, predicted, labels=[0, 1, 2], average="macro")),
        "qwk": float(cohen_kappa_score(truth, predicted, weights="quadratic")),
        "accuracy": float(accuracy_score(truth, predicted)),
        "per_class_f1": f1_score(truth, predicted, labels=[0, 1, 2], average=None).tolist(),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1, 2]).tolist(),
    }


def numeric_model(columns: list[str], config: dict, linear: bool = False):
    steps = []
    if linear:
        steps.append(("impute", SimpleImputer(strategy="median", keep_empty_features=True)))
    steps.append(("scale", StandardScaler()))
    prep = ColumnTransformer([
        ("numeric", Pipeline(steps), columns),
        ("content_type", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                                        sparse_output=False), ["content_type"]),
    ], sparse_threshold=0)
    if linear:
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                                 random_state=config["seed"])
    else:
        settings = {"class_weight": "balanced", "random_state": config["seed"],
                    **config["hgb"]}
        clf = HistGradientBoostingClassifier(**settings)
    return Pipeline([("prep", prep), ("clf", clf)])


def tabular_recipes(groups: dict) -> dict:
    variants = {
        "E0_baseline": ["baseline"],
        "E1a_recent5": ["baseline", "recent5"],
        "E1b_recent10": ["baseline", "recent5", "recent_extra"],
        "E1c_rhythm": ["baseline", "recent5", "recent_extra", "gap"],
        "E2a_style": ["baseline", "style"],
        "E2b_mentions": ["baseline", "mentions"],
        "E2_all": list(groups),
    }
    return {name: {"kind": "tabular", "columns": list(dict.fromkeys(
        column for group in selected for column in groups[group]))}
        for name, selected in variants.items()}


class FeatureContext:
    """All learned text transforms belong to this specific training partition."""
    def __init__(self, table, history, fit_indices, config, dense=None):
        self.table, self.history = table, history
        self.fit_indices, self.config, self.dense = fit_indices, config, dense
        self.svd = None
        self.vectors = {}
        self.history_features = {}
        lookup = pd.MultiIndex.from_frame(history[KEYS])
        self.history_positions = lookup.get_indexer(pd.MultiIndex.from_frame(table[KEYS]))
        if (self.history_positions < 0).any() or lookup.has_duplicates:
            raise ValueError("Post keys do not map uniquely to own-post history")

    def vector_values(self, source):
        if source not in self.vectors:
            if source == "svd":
                started = time.perf_counter()
                self.svd = fit_svd(self.table.iloc[self.fit_indices].content_text,
                                   self.config["text"])
                values = self.svd.transform(self.history.content_text)
                print(f"  fitted TF-IDF/SVD on training partition ({time.perf_counter()-started:.1f}s)",
                      flush=True)
            elif source == "e5" and self.dense is not None:
                values = self.dense
            else:
                raise ValueError(f"Missing representation: {source}")
            self.vectors[source] = np.asarray(values, dtype=np.float32)
        return self.vectors[source]

    def inputs(self, recipe):
        if recipe["kind"] == "text":
            return self.table[recipe["text_column"]], []
        columns = recipe["columns"].copy()
        frame = self.table[KEYS + ["content_type"] + columns].copy()
        kind = recipe["kind"]
        if kind in ("vector", "history"):
            source = recipe["source"]
            values = self.vector_values(source)
            if kind == "vector":
                extra = [f"{source}_{i:03d}" for i in range(values.shape[1])]
                added = pd.DataFrame(values[self.history_positions], columns=extra,
                                     index=self.table.index)
                frame = pd.concat([frame, added], axis=1)
            else:
                if source not in self.history_features:
                    self.history_features[source] = history_vector_features(
                        self.history, values, k=3, prefix=source)
                history_frame = self.history_features[source]
                similarity, outcomes = history_feature_columns(source)
                extra = similarity + (outcomes if recipe.get("outcomes") else [])
                frame = frame.merge(history_frame[KEYS + extra], on=KEYS, how="left",
                                    sort=False, validate="one_to_one")
                if not frame[KEYS].equals(self.table[KEYS]):
                    raise ValueError("Historical feature join changed row order")
            columns += extra
        return frame, columns


def fit_predict(recipe, context, fit_indices, prediction_indices):
    features, columns = context.inputs(recipe)
    if recipe["kind"] == "text":
        model = make_text_model(context.config["text"])
    else:
        model = numeric_model(columns, context.config, recipe.get("linear", False))
    y = context.table.iloc[fit_indices].label.astype(int)
    if set(y) != {0, 1, 2}:
        raise ValueError("Each training partition must contain all three classes")
    model.fit(features.iloc[fit_indices], y)
    if list(model.classes_) != [0, 1, 2]:
        raise ValueError("Unexpected probability column order")
    probabilities = model.predict_proba(features.iloc[prediction_indices])
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite predictions")
    return probabilities, model


def fingerprint(config, data_report):
    sources = [ROOT / "baseline.py", ROOT / "encode_text.py"]
    sources += sorted((ROOT / "kairos").glob("*.py"))
    return {
        "config": config, "data": data_report,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources if p.is_file()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "age_missing_policy": "raw NaN; HGB handles missing; linear imputer fit on training only",
    }


def partition(table, config):
    train_idx = np.flatnonzero(table.split.eq("train"))
    train = table.iloc[train_idx]
    fit, val = next(GroupShuffleSplit(n_splits=1, test_size=config["validation_fraction"],
                                      random_state=config["seed"]).split(train, groups=train.author))
    return train_idx[fit], train_idx[val]


def validate(table, history, groups, config, output, manifest, dense=None):
    fit_idx, val_idx = partition(table, config)
    print(f"fit: {len(fit_idx):,} posts / {table.iloc[fit_idx].author.nunique():,} authors; "
          f"validation: {len(val_idx):,} / {table.iloc[val_idx].author.nunique():,}", flush=True)
    fit_authors, val_authors = set(table.iloc[fit_idx].author), set(table.iloc[val_idx].author)
    if fit_authors & val_authors:
        raise ValueError("Overlapping validation authors")
    split_info = {"fit_authors": sorted(fit_authors), "validation_authors": sorted(val_authors),
                  "seed": config["seed"], "n_fit": len(fit_idx), "n_validation": len(val_idx)}
    save_json(output / "split.json", split_info)
    context = FeatureContext(table, history, fit_idx, config, dense)
    recipes = tabular_recipes(groups)
    records, probabilities = {}, {}
    y = table.iloc[val_idx].label.astype(int).to_numpy()
    run_hash = digest(manifest)
    (output / "validation_probabilities").mkdir(exist_ok=True, parents=True)

    def run(name, recipe):
        started = time.perf_counter()
        cache = output / "validation_probabilities" / f"{name}.npz"
        tag = digest({"run": run_hash, "recipe": recipe, "split": split_info})
        p = None
        if cache.exists():
            try:
                with cache.open("rb") as stream, np.load(stream, allow_pickle=False) as saved:
                    candidate = saved["probabilities"]
                    if (str(saved["tag"].item()) == tag
                            and candidate.shape == (len(val_idx), 3)
                            and candidate.dtype.kind in "fiu"
                            and np.isfinite(candidate).all()
                            and ((candidate >= 0) & (candidate <= 1)).all()
                            and np.allclose(candidate.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6)):
                        p = candidate
            except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile, zlib.error):
                pass  # An incomplete or invalid cache is a miss, not a prediction.
        if p is None:
            print(f"Fitting {name}...", flush=True)
            p, _ = fit_predict(recipe, context, fit_idx, val_idx)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=cache.parent, prefix=f".{name}-",
                                                 suffix=".npz", delete=False) as stream:
                    temporary = Path(stream.name)
                    np.savez_compressed(stream, probabilities=p, tag=tag)
                temporary.replace(cache)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        probabilities[name] = p
        records[name] = {**metrics(y, p.argmax(axis=1)),
                         "seconds": round(time.perf_counter() - started, 3), "recipe": recipe}
        print(f"{name:30s} macro-F1 {records[name]['macro_f1']:.5f} "
              f"QWK {records[name]['qwk']:.5f} ({records[name]['seconds']:.1f}s)", flush=True)
        save_json(output / "progress.json", records)

    for name, recipe in recipes.items():
        run(name, recipe)
    best_table = max(recipes, key=lambda name: records[name]["macro_f1"])
    best_columns = recipes[best_table]["columns"]
    if config.get("text_enabled", True):
        recipes.update({
            "E3a_text_main": {"kind": "text", "text_column": "text"},
            "E3b_text_enriched": {"kind": "text", "text_column": "content_text"},
            "E3c_svd_tabular": {"kind": "vector", "source": "svd", "columns": best_columns},
            "E3d_svd_history": {"kind": "history", "source": "svd", "columns": best_columns},
            "E3e_svd_neighbors": {"kind": "history", "source": "svd", "columns": best_columns,
                                   "outcomes": True},
        })
    if dense is not None:
        recipes.update({
            "E4a_e5_linear": {"kind": "vector", "source": "e5", "columns": best_columns,
                              "linear": True},
            "E4a_e5_hgb": {"kind": "vector", "source": "e5", "columns": best_columns},
            "E4b_e5_history": {"kind": "history", "source": "e5", "columns": best_columns},
            "E4c_e5_neighbors": {"kind": "history", "source": "e5", "columns": best_columns,
                                  "outcomes": True},
        })
    for name, recipe in recipes.items():
        if name not in records:
            run(name, recipe)
    weights = {name: {name: 1.0} for name in recipes}
    non_tabular = [name for name in recipes if recipes[name]["kind"] != "tabular"]
    for name in non_tabular:
        for w in config.get("blend_weights", [0.25, 0.5, 0.75]):
            blend_name = f"blend_{best_table}_{name}_{w}"
            p = (1 - w) * probabilities[best_table] + w * probabilities[name]
            weights[blend_name] = {best_table: 1-w, name: w}
            records[blend_name] = {**metrics(y, p.argmax(axis=1)), "seconds": 0,
                                   "weights": weights[blend_name]}
    ranked = sorted(records, key=lambda name: records[name]["macro_f1"], reverse=True)
    report = {"fingerprint": manifest, "run_sha256": run_hash, "split": split_info,
              "recipes": recipes, "results": records, "candidate_weights": weights,
              "best_tabular": best_table, "ranking": ranked, "selected": ranked[0],
              "test_labels_used_for_selection": False}
    save_json(output / "validation.json", report)
    pd.DataFrame([{"experiment": name, **{k: records[name][k] for k in
                 ("macro_f1", "qwk", "accuracy", "seconds")}} for name in ranked]).to_csv(
                     output / "experiments.csv", index=False)
    blended = sum(w * probabilities[name] for name, w in weights[ranked[0]].items())
    table.iloc[val_idx][KEYS].assign(label=blended.argmax(axis=1)).to_csv(
        output / "validation_predictions.csv", index=False)
    print(f"Validation selected: {ranked[0]} ({records[ranked[0]]['macro_f1']:.5f})", flush=True)
    return report


def stability(table, history, config, report, output, dense=None):
    nfold = config.get("cv_folds", 3)
    if nfold < 2:
        return {"selected": report["selected"], "skipped": True}
    candidates = list(dict.fromkeys(report["ranking"][:2] + ["E0_baseline"]))
    train_idx = np.flatnonzero(table.split.eq("train"))
    train = table.iloc[train_idx]
    oof = {name: np.zeros((len(train), 3), dtype=np.float32) for name in candidates}
    fold_metrics, fold_authors = {}, {}
    for fold, (fit_rel, val_rel) in enumerate(GroupKFold(n_splits=nfold).split(train, groups=train.author)):
        print(f"Cross-validation fold {fold+1}/{nfold}...", flush=True)
        fit_idx, val_idx = train_idx[fit_rel], train_idx[val_rel]
        context = FeatureContext(table, history, fit_idx, config, dense)
        required = {name for candidate in candidates
                    for name in report["candidate_weights"][candidate]}
        predictions = {}
        for name in sorted(required):
            p, _ = fit_predict(report["recipes"][name], context, fit_idx, val_idx)
            predictions[name] = p
        fold_metrics[str(fold)] = {}
        fold_authors[str(fold)] = sorted(table.iloc[val_idx].author.unique())
        for candidate in candidates:
            p = sum(w * predictions[name] for name, w in report["candidate_weights"][candidate].items())
            oof[candidate][val_rel] = p
            fold_metrics[str(fold)][candidate] = metrics(train.iloc[val_rel].label.astype(int), p.argmax(axis=1))
        save_json(output / "cv_progress.json", fold_metrics)
    scored = {name: metrics(train.label.astype(int), p.argmax(axis=1)) for name, p in oof.items()}
    selected = max(scored, key=lambda name: scored[name]["macro_f1"])
    uncertainty = paired_author_bootstrap(
        train.label.astype(int).to_numpy(), oof[selected].argmax(axis=1),
        oof["E0_baseline"].argmax(axis=1), train.author.to_numpy(),
        n_resamples=config.get("bootstrap_iterations", 1000), seed=config["seed"])
    np.savez_compressed(output / "oof_probabilities.npz", **oof)
    result = {"candidates": candidates, "fold_authors": fold_authors, "fold_metrics": fold_metrics,
              "pooled_metrics": scored, "selected": selected,
              "paired_author_bootstrap_vs_baseline": uncertainty,
              "interpretation": "Stability after model selection, not an independent final estimate"}
    save_json(output / "cross_validation.json", result)
    print(f"Cross-validation selected: {selected}, macro-F1 {scored[selected]['macro_f1']:.5f}", flush=True)
    return result


def final(table, history, config, report, selected, output, dense=None):
    fit_idx = np.flatnonzero(table.split.eq("train"))
    test_idx = np.flatnonzero(table.split.eq("test"))
    if table.iloc[test_idx].label.notna().any():
        raise ValueError("Prediction table must not expose test labels")
    context = FeatureContext(table, history, fit_idx, config, dense)
    weights = report["candidate_weights"][selected]
    predictions, models = {}, {}
    for name in weights:
        print(f"Final refit: {name} on {len(fit_idx):,} posts...", flush=True)
        predictions[name], models[name] = fit_predict(report["recipes"][name], context, fit_idx, test_idx)
    p = sum(weights[name] * predictions[name] for name in weights)
    frame = table.iloc[test_idx][KEYS].assign(label=p.argmax(axis=1))
    if frame.duplicated(KEYS).any() or not frame.label.isin([0, 1, 2]).all():
        raise ValueError("Invalid final predictions")
    target = output / "final"
    target.mkdir(exist_ok=True, parents=True)
    frame.to_csv(target / "predictions.csv", index=False)
    np.save(target / "probabilities.npy", p, allow_pickle=False)
    joblib.dump({"models": models, "recipes": {name: report["recipes"][name] for name in weights},
                 "svd": context.svd, "weights": weights, "config": config}, target / "model.joblib")
    save_json(target / "manifest.json", {
        "fingerprint": report["fingerprint"], "selected": selected, "weights": weights,
        "n_train": len(fit_idx), "n_test": len(test_idx),
        "test_labels_used_for_fitting_or_selection": False,
        "predictions_sha256": hashlib.sha256((target / "predictions.csv").read_bytes()).hexdigest(),
    })
    print(f"Saved {len(frame):,} predictions: {target / 'predictions.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cpu.json")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=["validate", "final", "all"], default="all")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output or ROOT / config["output"]
    if config.get("stage2"):
        if args.stage != "all":
            parser.error("Fixed stage2 pipeline supports --stage all only")
        from kairos.experiment_stage2 import reproduce
        reproduce(args.data, output, config)
        return
    output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=config.get("threads", 4)):
        table, history, groups, data_report = load_experiment_data(args.data)
        table = table.reset_index(drop=True)
        history = history.reset_index(drop=True)
        save_json(output / "data_report.json", data_report)
        dense, encoder_signature = None, None
        if config.get("encoder", {}).get("enabled"):
            from kairos.experiment_encode import encode_history
            encoder_config = {k: v for k, v in config["encoder"].items() if k != "enabled"}
            dense, metadata = encode_history(history, args.data / "cache/e5", encoder_config)
            save_json(output / "encoder_metadata.json", metadata)
            encoder_signature = {key: metadata[key] for key in (
                "model", "revision", "dimensions", "input", "config", "runtime",
                "encoder_sha256", "output_sha256")}
        manifest = fingerprint(config, data_report)
        manifest["encoder"] = encoder_signature
        save_json(output / "run_manifest.json", manifest)
        if args.stage in ("validate", "all"):
            report = validate(table, history, groups, config, output, manifest, dense)
        else:
            report = json.loads((output / "validation.json").read_text())
            if report["fingerprint"] != manifest:
                raise ValueError("Validation data, sources, versions or config changed; validate again")
        if args.stage in ("final", "all"):
            checked = stability(table, history, config, report, output, dense)
            final(table, history, config, report, checked["selected"], output, dense)


if __name__ == "__main__":
    main()
