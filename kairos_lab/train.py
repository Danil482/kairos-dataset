"""Author-grouped OOF model selection, persistent folds, and final predictions."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from baseline import model as baseline_model
from kairos.metrics import score
from .common import atomic_json, digest, identity, mark, stop_check, verified_marker
from .dataset import BASE_NUM, KEYS, release_identity
from .encoders import contextual_embeddings


class BudgetReached(RuntimeError):
    pass


def class_weights(y, power):
    counts = np.bincount(np.asarray(y, dtype=int), minlength=3)
    if (counts == 0).any():
        raise ValueError("All three classes required")
    weights = (len(y) / (3 * counts)) ** power
    return weights / weights.mean()


def estimator(spec, config, numeric, y, emb_columns=()):
    if spec["family"] == "baseline":
        result = baseline_model()
        if "baseline_iterations" in config:
            result.set_params(clf__max_iter=config["baseline_iterations"])
        return result
    base = [("num", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["content_type"])]
    if spec["family"] == "tfidf":
        settings = config["tfidf"]
        base[0] = ("num", Pipeline([("impute", SimpleImputer(keep_empty_features=True)),
                                   ("scale", StandardScaler())]), numeric)
        base += [("word", TfidfVectorizer(ngram_range=(1, 2), min_df=settings["min_df"],
                     max_features=settings["word_features"], sublinear_tf=True, dtype=np.float32), "model_text"),
                 ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=settings["min_df"],
                     max_features=settings["char_features"], sublinear_tf=True, dtype=np.float32), "model_text")]
        return Pipeline([("prep", ColumnTransformer(base, sparse_threshold=1.0)),
                         ("clf", LogisticRegression(C=spec["C"], solver="saga", max_iter=400,
                          random_state=config["seed"], class_weight="balanced", n_jobs=1, warm_start=True))])
    if emb_columns:
        # Fitted separately inside every fold, never on validation/test embeddings.
        n_components = min(64, len(emb_columns), len(y) - 1)
        base.append(("emb", PCA(n_components=n_components, random_state=config["seed"], svd_solver="randomized"), list(emb_columns)))
    params = spec["parameters"]
    classifier = CatBoostClassifier(iterations=params["iterations"], depth=params["depth"],
                   learning_rate=params["learning_rate"], loss_function="MultiClass", l2_leaf_reg=5,
                   random_seed=config["seed"], class_weights=class_weights(y, params["class_power"]).tolist(),
                   thread_count=config["threads"], task_type="CPU", verbose=False, allow_writing_files=True)
    return Pipeline([("prep", ColumnTransformer(base)), ("clf", classifier)])


def check_probabilities(values, rows):
    if values.shape != (rows, 3) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid probability matrix")
    if not np.allclose(values.sum(1), 1, atol=1e-5):
        raise ValueError("Probabilities do not sum to one")


def atomic_model(path, value):
    tmp = path.with_suffix(".tmp")
    joblib.dump(value, tmp)
    tmp.replace(path)


def fit_checkpointed(clf, spec, frame, y, folder, job, work, fingerprint):
    """Restore CatBoost's optimizer or a committed sparse optimizer chunk."""
    checkpoint = folder / "checkpoints" / job
    checkpoint.mkdir(parents=True, exist_ok=True)
    if spec["family"] not in ("baseline", "tfidf"):
        clf.set_params(clf__train_dir=str(checkpoint))
        clf.fit(frame, y, clf__save_snapshot=True,
                clf__snapshot_file=str(checkpoint / "optimizer.cbsnapshot"), clf__snapshot_interval=60)
        return clf
    if spec["family"] == "baseline":
        # The reference fits in seconds; preserve its exact fitting procedure.
        return clf.fit(frame, y)
    marker = checkpoint / "latest.json"
    completed = 0
    if marker.exists():
        if not verified_marker(marker, fingerprint):
            raise ValueError("Sparse checkpoint identity mismatch")
        state = json.loads(marker.read_text())
        clf = joblib.load(checkpoint / state["model"])
        completed = state["iterations"]
        if state.get("converged"):
            return clf
        matrix = clf.named_steps["prep"].transform(frame)
    else:
        matrix = clf.named_steps["prep"].fit_transform(frame, y)
    # Versioned commits preserve the previous valid optimizer if a write is interrupted.
    while completed < 400:
        stop_check(work)
        amount = min(50, 400 - completed)
        clf.named_steps["clf"].set_params(max_iter=amount)
        clf.named_steps["clf"].fit(matrix, y)
        completed += amount
        path = checkpoint / f"optimizer-{completed:04d}.joblib"
        atomic_model(path, clf)
        converged = int(np.max(clf.named_steps["clf"].n_iter_)) < amount
        mark(marker, fingerprint, [path], model=path.name, iterations=completed, converged=converged)
        atomic_json(work / "training-status.json", {"candidate": spec["name"], "job": job,
                                                    "stage": "fit", "optimizer_iterations": completed})
        if converged:
            break
    return clf


def evaluate_candidate(spec, config, train, test, numeric, work, budget_end, emb_columns=()):
    name = spec["name"]
    folder = work / "models" / name
    folder.mkdir(parents=True, exist_ok=True)
    fp = identity({"spec": spec, "config": config, "train": digest(work / "cache/train.parquet"),
                   "test": digest(work / "cache/test.parquet"), "numeric": numeric,
                   "embedding_columns": list(emb_columns)})
    marker = folder / "complete.json"
    if verified_marker(marker, fp):
        return np.load(folder / "probabilities.npz", allow_pickle=False)
    folds = sorted(train.fold.unique())
    oof = np.zeros((len(train), 3), dtype=np.float64)
    fold_metrics = []
    for fold in folds:
        stop_check(work)
        valid_idx = np.flatnonzero(train.fold.to_numpy() == fold)
        fit_idx = np.flatnonzero(train.fold.to_numpy() != fold)
        fold_fp = identity({"parent": fp, "fold": int(fold)})
        done = folder / f"fold-{fold}.json"
        prediction_path = folder / f"fold-{fold}.npz"
        if verified_marker(done, fold_fp):
            values = np.load(prediction_path, allow_pickle=False)["probabilities"]
        else:
            if time.time() >= budget_end:
                raise BudgetReached("Model-selection budget reached; completed candidates retained")
            atomic_json(work / "training-status.json", {"candidate": name, "fold": int(fold), "stage": "fit"})
            clf = estimator(spec, config, numeric, train.iloc[fit_idx].label, emb_columns)
            started = time.monotonic()
            clf = fit_checkpointed(clf, spec, train.iloc[fit_idx], train.iloc[fit_idx].label,
                                   folder, f"fold-{fold}", work, fold_fp)
            values = clf.predict_proba(train.iloc[valid_idx])
            check_probabilities(values, len(valid_idx))
            tmp = prediction_path.with_suffix(".tmp")
            with tmp.open("wb") as stream:
                np.savez_compressed(stream, probabilities=values)
            tmp.replace(prediction_path)
            atomic_model(folder / f"fold-{fold}.joblib", clf)
            mark(done, fold_fp, [prediction_path, folder / f"fold-{fold}.joblib"],
                 elapsed_seconds=time.monotonic() - started)
        check_probabilities(values, len(valid_idx))
        oof[valid_idx] = values
        fold_metrics.append(score(train.iloc[valid_idx].label, values.argmax(1)))
    stop_check(work)
    # Full fit is needed for a usable provisional result, even when the search budget just expired.
    atomic_json(work / "training-status.json", {"candidate": name, "stage": "full_fit"})
    clf = estimator(spec, config, numeric, train.label, emb_columns)
    clf = fit_checkpointed(clf, spec, train, train.label, folder, "full", work, fp)
    predicted = clf.predict_proba(test)
    check_probabilities(predicted, len(test))
    atomic_model(folder / "model.joblib", clf)
    probability_path = folder / "probabilities.npz"
    with probability_path.with_suffix(".tmp").open("wb") as stream:
        np.savez_compressed(stream, oof=oof, test=predicted)
    probability_path.with_suffix(".tmp").replace(probability_path)
    atomic_json(folder / "metrics.json", {"selection_split": "train_author_oof", "spec": spec,
                "oof": score(train.label, oof.argmax(1)), "folds": fold_metrics})
    mark(marker, fp, [folder / "model.joblib", probability_path, folder / "metrics.json"])
    return np.load(probability_path, allow_pickle=False)


def choose_blend(names: list[str], predictions: dict, y: np.ndarray):
    """Small deterministic probability search on train OOF only; no trained stacker."""
    def metric(p):
        return f1_score(y, p.argmax(1), average="macro", labels=[0, 1, 2])

    best = max(names, key=lambda n: metric(predictions[n]["oof"]))
    weights = {n: float(n == best) for n in names}
    current = predictions[best]["oof"].copy()
    current_score = metric(current)
    for _ in range(3):
        improved = None
        for name in names:
            for alpha in (.2, .4):
                candidate = (1 - alpha) * current + alpha * predictions[name]["oof"]
                value = metric(candidate)
                if value > current_score + .0005:
                    improved = (name, alpha, candidate, value)
                    current_score = value
        if improved is None:
            break
        name, alpha, current, current_score = improved
        weights = {n: w * (1 - alpha) + (alpha if n == name else 0) for n, w in weights.items()}
    scales = np.ones(3)
    best_score = metric(current)
    for below in (.85, 1.0, 1.15):
        for above in (.85, 1.0, 1.15):
            scale = np.array([below, 1.0, above])
            candidate_score = metric(current * scale)
            if candidate_score > best_score + .0005:
                scales, best_score = scale, candidate_score
    oof = sum(weights[n] * predictions[n]["oof"] for n in names)
    test = sum(weights[n] * predictions[n]["test"] for n in names)
    return weights, scales, oof, test


def write_prediction(path: Path, keys: pd.DataFrame, labels: np.ndarray):
    result = keys[KEYS].copy()
    result["label"] = labels
    if result.duplicated(KEYS).any() or not result.label.isin([0, 1, 2]).all():
        raise ValueError("Invalid final prediction keys/labels")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    result.to_csv(tmp, index=False)
    tmp.replace(path)


def publish_available(work, train, test, predictions):
    names = sorted(predictions)
    weights, scales, oof, values = choose_blend(names, predictions, train.label.to_numpy())
    write_prediction(work / "predictions/best_available.csv", test, (values * scales).argmax(1))
    selected = score(train.label, (oof * scales).argmax(1))
    atomic_json(work / "selection.json", {"partition": "train_author_oof", "weights": weights,
                "class_scales": scales.tolist(), "metrics": selected,
                "note": "Blend and class scales are selected on these OOF targets; this is a selection estimate, not an untouched holdout."})
    return oof, values, scales


def diagnostics(train, predicted, baseline, samples, seed):
    y = train.label.to_numpy()
    groups = {}
    masks = {f"type:{kind}": train.content_type.eq(kind).to_numpy() for kind in train.content_type.unique()}
    masks.update({"history:5-9": train.history_count.between(5, 9).to_numpy(),
                  "history:10-19": train.history_count.between(10, 19).to_numpy(),
                  "history:20+": train.history_count.ge(20).to_numpy(),
                  "text:cyrillic": train.cyrillic_fraction.ge(.2).to_numpy(),
                  "text:other": train.cyrillic_fraction.lt(.2).to_numpy(),
                  "media:document": train.has_document.eq(1).to_numpy()})
    for name, mask in masks.items():
        if mask.sum() >= 10:
            groups[name] = {"rows": int(mask.sum()), "macro_f1": float(f1_score(y[mask], predicted[mask], labels=[0, 1, 2], average="macro", zero_division=0))}
    authors, inverse = np.unique(train.author, return_inverse=True)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        weights = np.bincount(rng.integers(0, len(authors), len(authors)), minlength=len(authors))[inverse]
        value = f1_score(y, predicted, labels=[0, 1, 2], average="macro", sample_weight=weights, zero_division=0)
        reference = f1_score(y, baseline, labels=[0, 1, 2], average="macro", sample_weight=weights, zero_division=0)
        estimates.append([value, value - reference])
    intervals = np.quantile(estimates, [.025, .975], axis=0).tolist() if estimates else []
    return {"groups_train_oof": groups, "author_bootstrap": {"samples": samples,
              "columns": ["macro_f1", "delta_vs_baseline"], "lower_upper_95": intervals,
              "limitation": "OOF blend was selected using the same train targets; intervals do not remove model-selection bias."}}


def train_models(work: Path, config: dict):
    cache = work / "cache"
    train, test = pd.read_parquet(cache / "train.parquet"), pd.read_parquet(cache / "test.parquet")
    if "label" in test or "class" in test or set(train.author) & set(test.author):
        raise ValueError("Test label or author boundary violation")
    cols = json.loads((cache / "feature_columns.json").read_text())
    numeric = cols["numeric"]
    for df in (train, test):
        df["model_text"] = df.text.fillna("") + "\n" + df.document_text.fillna("") + "\nPROFILE: " + df.profile_text.fillna("").str[:1000]
    baseline = pd.read_parquet(cache / "baseline.parquet").drop(columns="split")
    base_train = train[KEYS + ["label", "fold"]].merge(baseline, on=KEYS, validate="one_to_one")
    base_test = test[KEYS].merge(baseline, on=KEYS, validate="one_to_one")
    specs = [{"name": "baseline", "family": "baseline"}]
    specs += [{"name": f"history-{i}", "family": "history", "parameters": params} for i, params in enumerate(config["catboost"])]
    specs += [{"name": f"tfidf-{i}", "family": "tfidf", "C": c} for i, c in enumerate(config["tfidf"]["C"])]
    emb_columns = []
    if (cache / "text_embeddings.json").is_file():
        text_fp = identity({"settings": config["text_encoder"], "data": digest(cache / "all_posts.parquet")})
        verified_marker(cache / "text_embeddings.json", text_fp)
        posts = pd.read_parquet(cache / "all_posts.parquet")
        outcomes = pd.read_parquet(cache / "history_outcomes.parquet")
        if not outcomes[KEYS].equals(posts[KEYS]):
            raise ValueError("Historical outcomes have different row order")
        values = contextual_embeddings(posts, np.load(cache / "text_embeddings.npy", allow_pickle=False),
                                       outcomes.reactions.to_numpy(), work)
        emb_columns = [f"embedding_{i}" for i in range(values.shape[1])]
        embedding_frame = pd.DataFrame(values, columns=emb_columns)
        embedding_frame[KEYS] = posts[KEYS]
        train = train.merge(embedding_frame, on=KEYS, validate="one_to_one")
        test = test.merge(embedding_frame, on=KEYS, validate="one_to_one")
        specs += [{"name": f"semantic-{i}", "family": "semantic", "parameters": params} for i, params in enumerate(config["catboost"][:2])]
    image_columns = []
    if (cache / "image_embeddings.json").is_file():
        image_fp = identity({"settings": config["image_encoder"], "data": digest(cache / "all_posts.parquet")})
        verified_marker(cache / "image_embeddings.json", image_fp)
        posts = pd.read_parquet(cache / "all_posts.parquet")
        values = np.load(cache / "image_embeddings.npy", allow_pickle=False)
        image_columns = [f"image_embedding_{i}" for i in range(values.shape[1])]
        frame = pd.DataFrame(values, columns=image_columns)
        frame[KEYS] = posts[KEYS]
        train = train.merge(frame, on=KEYS, validate="one_to_one")
        test = test.merge(frame, on=KEYS, validate="one_to_one")
        specs.append({"name": "multimodal", "family": "multimodal", "parameters": config["catboost"][0]})
    state_path = work / "search-budget.json"
    if not state_path.exists():
        atomic_json(state_path, {"started": time.time(), "ends": time.time() + config["max_training_hours"] * 3600})
    budget_end = json.loads(state_path.read_text())["ends"]
    if config.get("deadline"):
        budget_end = min(budget_end, datetime.fromisoformat(config["deadline"]).timestamp() - config["final_buffer_hours"] * 3600)
    predictions = {}
    for spec in specs:
        stop_check(work)
        family = spec["family"]
        columns = emb_columns + image_columns if family == "multimodal" else emb_columns if family == "semantic" else []
        try:
            results = evaluate_candidate(spec, config, base_train if family == "baseline" else train,
                                         base_test if family == "baseline" else test,
                                         BASE_NUM if family == "baseline" else numeric, work, budget_end, columns)
        except BudgetReached:
            if not predictions:
                raise
            break
        predictions[spec["name"]] = {k: results[k].copy() for k in ("oof", "test")}
        results.close()
        publish_available(work, train, test, predictions)
    oof, values, scales = publish_available(work, train, test, predictions)
    write_prediction(work / "predictions/final.csv", test, (values * scales).argmax(1))
    train[KEYS + ["fold", "label"]].assign(prediction=(oof * scales).argmax(1)).to_csv(work / "predictions/oof.csv", index=False)
    with (work / "predictions/blend_probabilities.npz").open("wb") as stream:
        np.savez_compressed(stream, oof=oof, test=values)
    report = {"selection": json.loads((work / "selection.json").read_text()),
              "candidates": {name: json.loads((work / "models" / name / "metrics.json").read_text()) for name in predictions},
              "confusion_train_oof": confusion_matrix(train.label, (oof * scales).argmax(1), labels=[0, 1, 2]).tolist(),
              "per_class_train_oof": classification_report(train.label, (oof * scales).argmax(1), output_dict=True, zero_division=0)}
    report.update(diagnostics(train, (oof * scales).argmax(1), predictions["baseline"]["oof"].argmax(1),
                              config["bootstrap_samples"], config["seed"]))
    atomic_json(work / "report/training.json", report)
    lines = ["# Kairos experiment report", "", "Model selection uses author-grouped train OOF only.", "",
             "| Candidate | OOF macro-F1 | QWK |", "| --- | --- | --- |"]
    for name, result in report["candidates"].items():
        lines.append(f"| {name} | {result['oof']['macro_f1']:.4f} | {result['oof']['qwk']:.4f} |")
    lines += ["", "The blend/class-scale estimate is tuned on OOF and may be optimistic.",
              "Collection-time follower counts, career snapshots and reactions of earlier posts are retrospective features.",
              "No claim about real-time prediction is made. See official-test.json after the frozen final evaluation.", ""]
    skipped = work / "optional-images-skipped.json"
    if skipped.exists():
        atomic_json(work / "report/optional-images-skipped.json", json.loads(skipped.read_text()))
        lines += ["Image embeddings were skipped; see optional-images-skipped.json for the schema diagnostic.", ""]
    (work / "report/REPORT.md").write_text("\n".join(lines))


def score_final(data: Path, work: Path, config: dict):
    """Separate final-only boundary: this is the only modeling stage reading test targets."""
    from kairos.data import load_labels
    path = work / "predictions/final.csv"
    prepared = json.loads((work / "cache/prepared.json").read_text())
    if prepared["fingerprint"] != identity({"release": release_identity(data), "config": config}):
        raise ValueError("Release changed since feature preparation; final scoring refused")
    fingerprint = digest(path)
    marker = work / "report/final-scored.json"
    if verified_marker(marker, fingerprint):
        return
    if marker.exists():
        raise ValueError("Final test was already scored for different predictions; do not tune against it")
    truth = load_labels(data)
    truth = truth[truth.split == "test"]
    pred = pd.read_csv(path)
    both = truth.merge(pred, on=KEYS, how="outer", suffixes=("", "_pred"), indicator=True, validate="one_to_one")
    if len(pred) != config["expected"]["test"] or not both["_merge"].eq("both").all():
        raise ValueError("Final predictions must cover exactly the official test keys")
    if not both.label_pred.isin([0, 1, 2]).all():
        raise ValueError("Invalid predicted class")
    result = {"partition": "official_test", "rows": len(both), "prediction_sha256": fingerprint,
              "metrics": score(both.label, both.label_pred),
              "per_class": classification_report(both.label, both.label_pred, output_dict=True, zero_division=0),
              "confusion": confusion_matrix(both.label, both.label_pred, labels=[0, 1, 2]).tolist()}
    result["target_macro_f1"] = 0.70
    result["target_met"] = result["metrics"]["macro_f1"] >= 0.70
    with np.load(work / "models/baseline/probabilities.npz", allow_pickle=False) as baseline:
        result["baseline_on_same_test"] = score(both.label, baseline["test"].argmax(1))
    atomic_json(work / "report/official-test.json", result)
    mark(marker, fingerprint, [work / "report/official-test.json"])
