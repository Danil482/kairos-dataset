"""Supervised semantic classifiers and fusion, using the existing nested author folds."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "6")

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from train_maria import SEED, CATEGORY, metrics, bootstrap_authors
from scripts.lecture_models import split_by_plan, ordered_probabilities
from scripts.pilot_models import restrict_support
from scripts.calibrate_lecture_decisions import select_decision
from scripts.semantic_embeddings import OUT, SOURCE, fingerprint, save_json

NAMES = ["e5_logistic", "e5_mlp", "e5_history_mlp", "e5_history_catboost", "semantic_fusion"]


def load_inputs():
    manifest = json.loads((OUT / "embedding_manifest.json").read_text())
    if manifest["config"]["input_sha256"] != fingerprint(SOURCE):
        raise ValueError("Embedding cache and training table differ")
    payload = joblib.load(SOURCE)
    table = payload["table"].reset_index(drop=True)
    if not table[["author", "post"]].equals(pd.read_csv(OUT / "train_keys.csv")):
        raise ValueError("Embedding row order differs from feature table")
    unique = np.load(OUT / "unique_vectors.npy", mmap_mode="r")
    vectors = np.vstack([np.zeros((1, unique.shape[1]), dtype=np.float32), unique])
    with np.load(OUT / "post_vector_indices.npz") as mapping:
        own = vectors[mapping["text"] + 1]
        expanded = vectors[mapping["expanded_text"] + 1]
    if not np.isfinite(own).all() or not np.isfinite(expanded).all():
        raise ValueError("Invalid semantic vectors")
    return payload, table, own, expanded, manifest


def neural_network(dimensions):
    return nn.Sequential(nn.Linear(dimensions, 256), nn.ReLU(), nn.Dropout(.25),
                         nn.Linear(256, 64), nn.ReLU(), nn.Dropout(.15), nn.Linear(64, 3))


def predict_network(model, matrix, batch_size=2048):
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(matrix), batch_size):
            inputs = torch.as_tensor(np.asarray(matrix[start:start+batch_size]), dtype=torch.float32, device="cuda")
            result.append(model(inputs).softmax(1).cpu().numpy())
    return np.concatenate(result)


def fit_network(train_x, train_y, cal_x, cal_y, directory, name):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    # Float32 is sufficient for these small heads; the frozen E5 encoder is separate.
    model = neural_network(train_x.shape[1]).to("cuda")
    train_tensor = torch.as_tensor(train_x, dtype=torch.float32, device="cuda")
    target = torch.as_tensor(np.array(train_y, dtype=np.int64, copy=True), dtype=torch.long, device="cuda")
    counts = np.bincount(train_y, minlength=3)
    weights = len(train_y) / (3 * counts)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device="cuda"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    best_score, best_epoch, best_state, history = -1., 0, None, []
    for epoch in range(1, 81):
        model.train()
        indices = torch.randperm(len(train_x), device="cuda")
        loss_sum = 0.
        for start in range(0, len(indices), 512):
            chosen = indices[start:start+512]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(train_tensor[chosen]), target[chosen])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(chosen)
        cal_probability = predict_network(model, cal_x)
        value = float(f1_score(cal_y, cal_probability.argmax(1), average="macro", labels=[0, 1, 2]))
        history.append(dict(epoch=epoch, train_weighted_loss=loss_sum/len(train_x), inner_macro_f1=value))
        if value > best_score + 1e-5:
            best_score, best_epoch = value, epoch
            best_state = {key:weight.detach().cpu().clone() for key,weight in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f"{name}: epoch {epoch}, inner macro-F1 {value:.4f}, best epoch {best_epoch}", flush=True)
        if epoch-best_epoch >= 12:
            break
    model.load_state_dict(best_state)
    model.eval()
    torch.save(dict(state_dict=best_state, dimensions=train_x.shape[1], class_order=[0, 1, 2]), directory / f"{name}.pt")
    save_json(directory / f"{name}_learning_curve.json", dict(history=history, best_epoch=best_epoch,
              selection="Raw macro-F1 on inner calibration authors only", seed=SEED))
    return model, dict(best_epoch=best_epoch, epochs=len(history), best_inner_macro_f1=best_score,
                       layers=[train_x.shape[1], 256, 64, 3], class_weights=weights.tolist())


def tabular_preprocessor(numeric, cats):
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
    ])


def record_predictions(name, cal, val, calibration, validation, low, report, predictions, probabilities, directory,
                       started, details=None, config=None):
    cal = restrict_support(cal, calibration, low)
    val = restrict_support(val, validation, low)
    decision = config or select_decision(calibration.label, cal)
    predicted = (val * decision["multipliers"]).argmax(1)
    predictions[name] = predicted
    probabilities[name] = (cal, val)
    report["models"][name] = dict(metrics=metrics(validation.label, predicted), calibration=decision,
                                   seconds=round(time.monotonic()-started, 1), details=details or {})
    np.savez_compressed(directory / f"{name}_probabilities.npz", calibration=cal, validation=val)
    predictions.to_csv(directory / "predictions.csv", index=False)
    save_json(directory / "report.json", report)
    print(f"Fold {report['fold']} {name}: macro-F1 {report['models'][name]['metrics']['macro_f1']:.4f}", flush=True)


def choose_fusion(calibration_labels, baseline, probabilities):
    best = None
    for name in NAMES[:-1]:
        cal = probabilities[name][0]
        for weight in (0., .1, .25, .5, .75, 1.):
            decision = select_decision(calibration_labels, (1-weight)*baseline + weight*cal)
            decision.update(semantic_model=name, semantic_weight=weight)
            if best is None or decision["calibration_macro_f1"] > best["calibration_macro_f1"]:
                best = decision
    return best


def train_fold(number, payload, table, own, expanded, manifest):
    directory = OUT / f"fold_{number}"
    directory.mkdir(exist_ok=True)
    report_path = directory / "report.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous.get("complete"):
            if previous["embedding_config"] != manifest["config"]:
                raise ValueError("Completed models use a different embedding cache")
            print(f"Fold {number} is already complete", flush=True)
            return
    plan = json.loads((ROOT / "results" / "maria" / f"fold_{number}.json").read_text())
    fitting, calibration, validation = split_by_plan(table, plan)
    fi, ci, vi = (part.index.to_numpy() for part in (fitting, calibration, validation))
    numeric = payload["numeric"] + payload["extra_numeric"]
    cats = CATEGORY + ["writing_system"]
    forbidden = {"author", "post", "label", "class", "split", "reactions", "resid", "age_days", "log_followers"}
    if forbidden.intersection(numeric+cats):
        raise ValueError("Invalid tabular feature allowlist")
    low = payload["cutoffs"][0]
    report = dict(fold=number, seed=SEED, embedding_config=manifest["config"], complete=False,
        fitting_posts=len(fitting), calibration_posts=len(calibration), validation_posts=len(validation),
        scope="Official train authors only; same nested author folds as previous experiments",
        author_plan={key:plan[key] for key in ("fitting_authors", "calibration_authors", "validation_authors")},
        models={})
    predictions = validation[["author", "post", "label"]].copy()
    calibration[["author", "post", "label"]].to_csv(directory / "calibration_keys.csv", index=False)
    probabilities = {}
    save_json(report_path, report)

    print(f"Fold {number}: fitting semantic logistic regression", flush=True)
    started, best, search = time.monotonic(), None, []
    for regularization in (.1, 1., 10.):
        model = LogisticRegression(C=regularization, class_weight="balanced", solver="lbfgs", max_iter=600, random_state=SEED)
        model.fit(expanded[fi], fitting.label)
        cal = ordered_probabilities(model, expanded[ci])
        decision = select_decision(calibration.label, restrict_support(cal, calibration, low))
        search.append(dict(C=regularization, calibration_macro_f1=decision["calibration_macro_f1"]))
        if best is None or decision["calibration_macro_f1"] > best["decision"]["calibration_macro_f1"]:
            best = dict(model=model, decision=decision, C=regularization, cal=cal)
    joblib.dump(best["model"], directory / "e5_logistic.joblib", compress=3)
    val = ordered_probabilities(best["model"], expanded[vi])
    reloaded = joblib.load(directory / "e5_logistic.joblib")
    np.testing.assert_allclose(reloaded.predict_proba(expanded[vi[:64]]), val[:64], atol=1e-8)
    record_predictions("e5_logistic", best["cal"], val, calibration, validation, low, report, predictions,
        probabilities, directory, started, dict(C=best["C"], inner_search=search, input="expanded E5", roundtrip_passed=True), best["decision"])
    del best, model, reloaded

    semantic = np.concatenate([own, expanded], axis=1)
    tabular = tabular_preprocessor(numeric, cats)
    tab_train = tabular.fit_transform(fitting)
    tab_cal = tabular.transform(calibration)
    tab_val = tabular.transform(validation)
    joblib.dump(tabular, directory / "tabular_preprocessor.joblib", compress=3)
    for name, hybrid in (("e5_mlp", False), ("e5_history_mlp", True)):
        print(f"Fold {number}: fitting {name}", flush=True)
        started = time.monotonic()
        scaler = StandardScaler().fit(semantic[fi])
        train_x, cal_x, val_x = (scaler.transform(semantic[index]).astype(np.float32) for index in (fi, ci, vi))
        if hybrid:
            train_x = np.concatenate([train_x, tab_train], axis=1)
            cal_x = np.concatenate([cal_x, tab_cal], axis=1)
            val_x = np.concatenate([val_x, tab_val], axis=1)
        train_x, cal_x, val_x = (np.clip(x, -8, 8).astype(np.float32) for x in (train_x, cal_x, val_x))
        joblib.dump(scaler, directory / f"{name}_scaler.joblib", compress=3)
        model, details = fit_network(train_x, fitting.label.to_numpy(), cal_x, calibration.label.to_numpy(), directory, name)
        cal, val = predict_network(model, cal_x), predict_network(model, val_x)
        saved = torch.load(directory / f"{name}.pt", weights_only=True, map_location="cpu")
        reloaded = neural_network(saved["dimensions"]).to("cuda")
        reloaded.load_state_dict(saved["state_dict"])
        np.testing.assert_allclose(predict_network(reloaded, val_x[:64]), val[:64], rtol=1e-5, atol=1e-6)
        details.update(roundtrip_passed=True, uses_history=hybrid, scaling="fit on fitting authors; clip to [-8,8]")
        record_predictions(name, cal, val, calibration, validation, low, report, predictions, probabilities, directory, started, details)
        del model, reloaded, train_x, cal_x, val_x
        torch.cuda.empty_cache()

    print(f"Fold {number}: fitting CatBoost with semantic PCA and history", flush=True)
    started = time.monotonic()
    pca = PCA(n_components=64, svd_solver="randomized", random_state=SEED)
    pca.fit(semantic[fi])
    pca_columns = [f"semantic_pca_{i}" for i in range(64)]
    transformed = []
    for part in (fitting, calibration, validation):
        matrix = part[numeric+cats].copy()
        projection = pca.transform(semantic[part.index])
        matrix[pca_columns] = projection
        transformed.append(matrix)
    model = CatBoostClassifier(iterations=800, depth=6, learning_rate=.045, loss_function="MultiClass",
        l2_leaf_reg=5, random_seed=SEED, thread_count=6, verbose=False, allow_writing_files=False)
    model.fit(transformed[0], fitting.label, cat_features=cats,
              eval_set=(transformed[1], calibration.label), early_stopping_rounds=100, use_best_model=True)
    model.save_model(str(directory / "e5_history_catboost.cbm"))
    joblib.dump(pca, directory / "semantic_pca.joblib", compress=3)
    save_json(directory / "feature_schema.json", dict(numeric=numeric, categorical=cats, pca_columns=pca_columns,
        cutoffs=payload["cutoffs"], mlp_clip=8, semantic_order=["own", "expanded"], class_order=[0, 1, 2]))
    cal, val = ordered_probabilities(model, transformed[1]), ordered_probabilities(model, transformed[2])
    reloaded = CatBoostClassifier().load_model(str(directory / "e5_history_catboost.cbm"))
    np.testing.assert_allclose(reloaded.predict_proba(transformed[2].iloc[:64]), val[:64], atol=1e-8)
    record_predictions("e5_history_catboost", cal, val, calibration, validation, low, report, predictions,
        probabilities, directory, started, dict(trees=int(model.tree_count_), pca_components=64,
        pca_explained_variance=float(pca.explained_variance_ratio_.sum()), roundtrip_passed=True))

    baseline_directory = ROOT / "results" / "lecture_models" / f"fold_{number}"
    for keys, filename in ((calibration, "calibration_keys.csv"), (validation, "predictions.csv")):
        expected = pd.read_csv(baseline_directory / filename)[["author", "post", "label"]]
        if not keys[["author", "post", "label"]].reset_index(drop=True).equals(expected):
            raise ValueError("Baseline probability row order differs")
    with np.load(baseline_directory / "catboost_800_probabilities.npz") as saved:
        baseline_cal, baseline_val = saved["calibration"], saved["validation"]
    started = time.monotonic()
    decision = choose_fusion(calibration.label, baseline_cal, probabilities)
    weight = decision["semantic_weight"]
    cal, val = probabilities[decision["semantic_model"]]
    record_predictions("semantic_fusion", (1-weight)*baseline_cal+weight*cal, (1-weight)*baseline_val+weight*val,
        calibration, validation, low, report, predictions, probabilities, directory, started,
        dict(selection="semantic model, fusion weight and class multipliers selected only on inner authors",
             weight_grid=[0., .1, .25, .5, .75, 1.]), decision)
    report["complete"] = True
    save_json(report_path, report)


def summarize(table):
    parts, reports = [], []
    for number in (1, 2, 3):
        directory = OUT / f"fold_{number}"
        if not (directory / "report.json").exists():
            return
        report = json.loads((directory / "report.json").read_text())
        if not report["complete"]:
            return
        reports.append(report)
        parts.append(pd.read_csv(directory / "predictions.csv").assign(fold=number))
    combined = pd.concat(parts, ignore_index=True)
    coverage = combined.merge(table[["author", "post", "label"]], on=["author", "post", "label"], how="outer", indicator=True, validate="one_to_one")
    if not coverage._merge.eq("both").all():
        raise ValueError("OOF predictions do not cover each official train post once")
    reference = pd.read_csv(ROOT / "results" / "lecture_models" / "decision_audit" / "oof_predictions.csv")
    combined = combined.merge(reference[["author", "post", "label", "catboost_800", "catboost_svm_fusion"]],
        on=["author", "post", "label"], how="left", validate="one_to_one")
    if combined.isna().any().any():
        raise ValueError("Missing baseline predictions")
    names = NAMES + ["catboost_800", "catboost_svm_fusion"]
    report = dict(n=len(combined), authors=int(combined.author.nunique()),
        scope="Development CV on official train authors; not an official test evaluation",
        oof={name:metrics(combined.label, combined[name]) for name in names},
        folds={str(r["fold"]):{name:r["models"][name]["metrics"]["macro_f1"] for name in NAMES} for r in reports},
        fusion_minus_previous_best_95=bootstrap_authors(combined, "semantic_fusion", "catboost_svm_fusion"),
        semantic_catboost_minus_previous_best_95=bootstrap_authors(combined, "e5_history_catboost", "catboost_svm_fusion"),
        fusion_minus_semantic_catboost_95=bootstrap_authors(combined, "semantic_fusion", "e5_history_catboost"),
        hybrid_mlp_minus_previous_best_95=bootstrap_authors(combined, "e5_history_mlp", "catboost_svm_fusion"),
        interpretation="Author bootstrap conditions on predictions; it does not include model-selection uncertainty.",
        all_saved_classifiers_roundtrip_passed=all(r["models"][name]["details"].get("roundtrip_passed") for r in reports for name in NAMES[:-1]))
    slices = combined.merge(table[["author", "post", "original_chars", "has_reshared", "hist_n"]], on=["author", "post"], validate="one_to_one")
    masks = dict(empty_original=slices.original_chars.eq(0), has_original=slices.original_chars.gt(0),
                 has_reshared=slices.has_reshared.eq(1), no_reshared=slices.has_reshared.eq(0),
                 short_history=slices.hist_n.lt(10))
    report["slices"] = {name:dict(n=int(mask.sum()), previous=metrics(slices.loc[mask, "label"], slices.loc[mask, "catboost_svm_fusion"]),
        semantic=metrics(slices.loc[mask, "label"], slices.loc[mask, "semantic_fusion"])) for name,mask in masks.items()}
    report["best_development_model"] = max(names, key=lambda name:report["oof"][name]["by_class"]["macro avg"]["f1-score"])
    combined.to_csv(OUT / "oof_predictions.csv", index=False)
    save_json(OUT / "comparison.json", report)
    print(json.dumps({"oof_macro_f1":{name:report["oof"][name]["macro_f1"] for name in names},
        "fusion_minus_previous_best_95":report["fusion_minus_previous_best_95"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=[1, 2, 3])
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this configured semantic experiment")
    payload, table, own, expanded, manifest = load_inputs()
    with threadpool_limits(limits=6):
        if not args.summarize_only:
            for number in ([args.fold] if args.fold else [1, 2, 3]):
                train_fold(number, payload, table, own, expanded, manifest)
        summarize(table)


if __name__ == "__main__":
    main()
