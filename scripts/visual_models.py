"""Controlled visual-content ablation on the existing nested author folds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from train_maria import SEED, CATEGORY, metrics, bootstrap_authors
from scripts.lecture_models import split_by_plan, ordered_probabilities
from scripts.semantic_models import (load_inputs, tabular_preprocessor, fit_network, predict_network,
                                    neural_network, record_predictions)
from scripts.semantic_embeddings import save_json
from scripts.calibrate_lecture_decisions import select_decision
from scripts.visual_embeddings import OUT
from scripts.audit_visual_inputs import overlap_by_fold

NAMES = ["media_metadata_catboost", "multimodal_catboost", "multimodal_mlp", "visual_fusion"]
REFERENCE = ROOT / "results" / "semantic"


def load_visual(table):
    manifest = json.loads((OUT / "aggregation.json").read_text())
    semantic_manifest = json.loads((REFERENCE / "embedding_manifest.json").read_text())
    if not manifest["complete"] or manifest["config"]["index"]["table_sha256"] != semantic_manifest["config"]["input_sha256"]:
        raise ValueError("Semantic and visual caches refer to different inputs")
    if not table[["author", "post"]].equals(pd.read_csv(OUT / "train_keys.csv")):
        raise ValueError("Visual vectors have a different row order")
    metadata = joblib.load(OUT / "metadata.joblib")
    visual = np.concatenate([np.load(OUT / "post_mean.npy"), np.load(OUT / "post_first.npy")], axis=1)
    if len(metadata) != len(table) or len(visual) != len(table) or not np.isfinite(visual).all():
        raise ValueError("Invalid visual feature matrix")
    return pd.concat([table, metadata], axis=1), visual, metadata.columns.tolist(), manifest


def image_projection(pca, vectors, available):
    projected = pca.transform(vectors).astype(np.float32)
    projected[~np.asarray(available, dtype=bool)] = 0
    return projected


def train_fold(number, payload, table, semantic, visual, media_columns, manifest):
    directory = OUT / f"fold_{number}"
    directory.mkdir(exist_ok=True)
    report_path = directory / "report.json"
    if report_path.exists():
        saved = json.loads(report_path.read_text())
        if saved.get("complete"):
            if saved["visual_config"] != manifest["config"]:
                raise ValueError("Existing models use another image cache")
            print(f"Visual fold {number} already complete", flush=True)
            return
    plan = json.loads((ROOT / "results" / "maria" / f"fold_{number}.json").read_text())
    parts = fitting, calibration, validation = split_by_plan(table, plan)
    fi, ci, vi = (part.index.to_numpy() for part in parts)
    baseline_directory = REFERENCE / f"fold_{number}"
    for keys, filename in ((calibration, "calibration_keys.csv"), (validation, "predictions.csv")):
        if not keys[["author", "post", "label"]].reset_index(drop=True).equals(pd.read_csv(baseline_directory / filename)[["author", "post", "label"]]):
            raise ValueError("Reference probabilities are misaligned")
    numeric = payload["numeric"] + payload["extra_numeric"] + media_columns
    cats = CATEGORY + ["writing_system"]
    if {"author", "post", "label", "reactions", "resid", "age_days", "log_followers"}.intersection(numeric+cats):
        raise ValueError("Outcome or snapshot feature entered the allowlist")
    report = dict(fold=number, complete=False, seed=SEED, visual_config=manifest["config"],
        scope="Development CV on official train only, fixed nested author folds", models={},
        author_plan={key:plan[key] for key in ("fitting_authors", "calibration_authors", "validation_authors")})
    predictions = validation[["author", "post", "label"]].copy()
    calibration[["author", "post", "label"]].to_csv(directory / "calibration_keys.csv", index=False)
    probabilities = {}
    low = payload["cutoffs"][0]
    # Reuse the exact semantic projection fitted on these fitting authors.
    pca = joblib.load(baseline_directory / "semantic_pca.joblib")
    joblib.dump(pca, directory / "semantic_pca.joblib", compress=3)
    semantic_columns = [f"semantic_pca_{i}" for i in range(64)]
    matrices = []
    for part in parts:
        matrices.append(pd.concat([part[numeric+cats].copy(),
            pd.DataFrame(pca.transform(semantic[part.index]), index=part.index, columns=semantic_columns)], axis=1))
    image_pca = PCA(n_components=64, svd_solver="randomized", random_state=SEED)
    image_pca.fit(visual[fi[fitting.visual_available.to_numpy(dtype=bool)]])
    joblib.dump(image_pca, directory / "visual_pca.joblib", compress=3)
    visual_columns = [f"visual_pca_{i}" for i in range(64)]
    projected = [pd.DataFrame(image_projection(image_pca, visual[part.index], part.visual_available),
                              index=part.index, columns=visual_columns) for part in parts]
    save_json(directory / "feature_schema.json", dict(numeric=numeric, categorical=cats, semantic_columns=semantic_columns,
        visual_columns=visual_columns, semantic_order=["own", "expanded"], visual_order=["mean", "first_manifest"],
        cutoffs=payload["cutoffs"], class_order=[0, 1, 2], mlp_clip=8))
    for name, with_images in ((NAMES[0], False), (NAMES[1], True)):
        print(f"Fold {number}: fitting {name}", flush=True)
        started = time.monotonic()
        inputs = [pd.concat([matrix, image], axis=1) for matrix,image in zip(matrices, projected)] if with_images else matrices
        model = CatBoostClassifier(iterations=800, depth=6, learning_rate=.045, loss_function="MultiClass",
            l2_leaf_reg=5, random_seed=SEED, thread_count=6, verbose=False, allow_writing_files=False)
        model.fit(inputs[0], fitting.label, cat_features=cats, eval_set=(inputs[1], calibration.label),
                  early_stopping_rounds=100, use_best_model=True)
        model.save_model(str(directory / f"{name}.cbm"))
        cal, val = ordered_probabilities(model, inputs[1]), ordered_probabilities(model, inputs[2])
        reloaded = CatBoostClassifier().load_model(str(directory / f"{name}.cbm"))
        np.testing.assert_allclose(reloaded.predict_proba(inputs[2].iloc[:64]), val[:64], atol=1e-8)
        importance = sorted(zip(inputs[0].columns, model.feature_importances_), key=lambda pair:-pair[1])
        feature_groups = {"media_metadata":set(media_columns), "semantic_pca":set(semantic_columns),
                          "visual_pca":set(visual_columns), "history_and_other":set(payload["numeric"]+payload["extra_numeric"]+cats)}
        record_predictions(name, cal, val, calibration, validation, low, report, predictions, probabilities,
            directory, started, dict(trees=int(model.tree_count_), roundtrip_passed=True, uses_image_content=with_images,
                visual_pca_variance=float(image_pca.explained_variance_ratio_.sum()) if with_images else None,
                feature_importance_by_group={group:float(sum(value for key,value in importance if key in columns))
                                             for group,columns in feature_groups.items()},
                top_features=[dict(feature=key, importance=float(value)) for key,value in importance[:25]]))

    name = "multimodal_mlp"
    print(f"Fold {number}: fitting {name}", flush=True)
    started = time.monotonic()
    raw = np.concatenate([semantic, visual], axis=1)
    scaler = StandardScaler().fit(raw[fi])
    tabular = tabular_preprocessor(numeric, cats).fit(fitting)
    joblib.dump(scaler, directory / "multimodal_scaler.joblib", compress=3)
    joblib.dump(tabular, directory / "tabular_preprocessor.joblib", compress=3)
    inputs = [np.clip(np.concatenate([scaler.transform(raw[part.index]), tabular.transform(part)], axis=1), -8, 8).astype(np.float32) for part in parts]
    model, details = fit_network(inputs[0], fitting.label.to_numpy(), inputs[1], calibration.label.to_numpy(), directory, name)
    cal, val = predict_network(model, inputs[1]), predict_network(model, inputs[2])
    saved = torch.load(directory / f"{name}.pt", weights_only=True, map_location="cpu")
    reloaded = neural_network(saved["dimensions"]).to("cuda")
    reloaded.load_state_dict(saved["state_dict"])
    np.testing.assert_allclose(predict_network(reloaded, inputs[2][:64]), val[:64], rtol=1e-5, atol=1e-6)
    details.update(roundtrip_passed=True, input_order=["E5 own", "E5 expanded", "SigLIP mean", "SigLIP first", "table"],
                   encoder_training="Frozen encoders; supervised classifier head is trained")
    record_predictions(name, cal, val, calibration, validation, low, report, predictions, probabilities, directory, started, details)
    del inputs, raw, model, reloaded
    torch.cuda.empty_cache()

    started, best = time.monotonic(), None
    with np.load(baseline_directory / "e5_history_catboost_probabilities.npz") as saved:
        baseline_cal, baseline_val = saved["calibration"], saved["validation"]
    # Fixed small grid. No choice uses outer validation labels.
    for name in NAMES[:-1]:
        for weight in (0., .1, .25, .5, .75, 1.):
            decision = select_decision(calibration.label, (1-weight)*baseline_cal+weight*probabilities[name][0])
            decision.update(visual_model=name, visual_weight=weight)
            if best is None or decision["calibration_macro_f1"] > best["calibration_macro_f1"]:
                best = decision
    weight = best["visual_weight"]
    cal, val = probabilities[best["visual_model"]]
    record_predictions("visual_fusion", (1-weight)*baseline_cal+weight*cal, (1-weight)*baseline_val+weight*val,
        calibration, validation, low, report, predictions, probabilities, directory, started,
        dict(selection="Model, weight and multipliers selected on inner calibration authors only",
             baseline="e5_history_catboost", weight_grid=[0., .1, .25, .5, .75, 1.]), best)
    report["complete"] = True
    save_json(report_path, report)


def summarize(table):
    reports, parts = [], []
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
        raise ValueError("OOF coverage is incomplete")
    refs = ["e5_history_catboost", "semantic_fusion"]
    combined = combined.merge(pd.read_csv(REFERENCE / "oof_predictions.csv")[["author", "post", "label"]+refs],
                              on=["author", "post", "label"], validate="one_to_one")
    if combined.isna().any().any() or len(combined) != len(table):
        raise ValueError("Missing reference predictions")
    names = refs+NAMES
    report = dict(n=len(combined), authors=int(combined.author.nunique()),
        scope="Development CV on official train; official test not evaluated",
        oof={name:metrics(combined.label, combined[name]) for name in names},
        folds={str(r["fold"]):{name:r["models"][name]["metrics"]["macro_f1"] for name in NAMES} for r in reports},
        comparisons_95={name+"_minus_"+reference:bootstrap_authors(combined, name, reference)
            for name,reference in [(NAMES[0], refs[0]), (NAMES[1], NAMES[0]), (NAMES[1], refs[1]), (NAMES[3], refs[1])]},
        interpretation="Paired author bootstrap conditional on these predictions, excluding model-selection uncertainty",
        all_saved_classifiers_roundtrip_passed=all(r["models"][name]["details"].get("roundtrip_passed") for r in reports for name in NAMES[:-1]))
    slices = combined.merge(table[["author", "post", "original_chars", "visual_available"]], on=["author", "post"], validate="one_to_one")
    masks = dict(has_images=slices.visual_available.eq(1), no_images=slices.visual_available.eq(0),
                 empty_original=slices.original_chars.eq(0), empty_original_with_images=slices.original_chars.eq(0)&slices.visual_available.eq(1))
    overlap, overlap_counts = overlap_by_fold(table, json.loads((OUT / "media_index.json").read_text()))
    seen = pd.Series([overlap[(row.author, row.post)] for row in slices.itertuples()], index=slices.index)
    masks.update(images_seen_on_fitting_authors=slices.visual_available.eq(1)&seen,
                 images_not_seen_on_fitting_authors=slices.visual_available.eq(1)&~seen)
    report["cross_author_image_overlap"] = overlap_counts
    report["slices"] = {name:dict(n=int(mask.sum()), models={model:metrics(slices.loc[mask, "label"], slices.loc[mask, model]) for model in names}) for name,mask in masks.items()}
    report["best_development_model"] = max(names, key=lambda name:report["oof"][name]["by_class"]["macro avg"]["f1-score"])
    combined.to_csv(OUT / "oof_predictions.csv", index=False)
    save_json(OUT / "comparison.json", report)
    print(json.dumps({"oof_macro_f1":{name:report["oof"][name]["macro_f1"] for name in names},
                      "comparisons_95":report["comparisons_95"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=[1, 2, 3])
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    payload, table, own, expanded, _ = load_inputs()
    table, visual, media_columns, manifest = load_visual(table)
    semantic = np.concatenate([own, expanded], axis=1)
    with threadpool_limits(limits=6):
        if not args.summarize_only:
            for number in ([args.fold] if args.fold else [1, 2, 3]):
                train_fold(number, payload, table, semantic, visual, media_columns, manifest)
        summarize(table)


if __name__ == "__main__":
    main()
