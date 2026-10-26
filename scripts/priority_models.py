"""Independent v2 ablations, ordinal classification and honest holdout blending."""
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
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from train_maria import SEED, CATEGORY, metrics, bootstrap_authors
from scripts.semantic_models import load_inputs, record_predictions
from scripts.lecture_models import split_by_plan, ordered_probabilities
from scripts.history_models import load_history, PARAMETERS
from scripts.visual_models import load_visual, image_projection
from scripts.semantic_embeddings import save_json, fingerprint
from scripts.calibrate_lecture_decisions import select_decision
from scripts.pilot_models import restrict_support
from scripts.priority_features import OUT, POLICY, HISTORY

FEATURE_NAMES = POLICY["feature_experiments"]
NAMES = FEATURE_NAMES + ["balanced_loss", "ordinal", "blend_text", "blend_visual", "constant_blend", "adaptive_blend"]


def ordinal_probabilities(lower, upper):
    """Euclidean monotone projection of P(y>0), P(y>1), then class masses."""
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if lower.shape != upper.shape or not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("Invalid ordinal probabilities")
    if np.any((lower<0)|(lower>1)|(upper<0)|(upper>1)):
        raise ValueError("Ordinal probabilities outside [0,1]")
    mean = (lower+upper)/2
    crossed = upper>lower
    lower, upper = np.where(crossed, mean, lower), np.where(crossed, mean, upper)
    return np.column_stack([1-lower, lower-upper, upper])


def blend_features(table):
    return np.column_stack([np.log1p(table.original_chars), np.log1p(table.extra_chars.clip(lower=0)),
        np.log1p(table.hist_n), table.has_reshared, table.has_document, table.has_video,
        table.visual_available, np.log1p(table.visual_count)])


def gate_objective(coef, x, first, second, labels, strength):
    rows = np.arange(len(labels))
    left, right = first[rows, labels], second[rows, labels]
    weight = expit(x @ coef)
    true_probability = np.maximum((1-weight)*left+weight*right, 1e-12)
    penalty = coef.copy()
    penalty[0] = 0
    loss = -np.log(true_probability).mean() + .5*strength*float(penalty @ penalty)
    gradient = x.T @ ((left-right)*weight*(1-weight)/true_probability)/len(labels) + strength*penalty
    return loss, gradient


def gate_matrix(table, mean, scale):
    return np.column_stack([np.ones(len(table)), (blend_features(table)-mean)/scale])


def split_gate_authors(calibration):
    authors = np.array(sorted(calibration.author.unique()))
    np.random.default_rng(SEED+71).shuffle(authors)
    fit_authors = set(authors[:len(authors)//2])
    fitting = calibration.author.isin(fit_authors).to_numpy()
    if not fitting.any() or fitting.all():
        raise ValueError("Need separate gate fit/selection authors")
    return fitting


def add_field_projection(parts, field_vectors, fitting, directory, field, dimensions):
    available = np.linalg.norm(field_vectors, axis=1)>1e-8
    training = fitting.index[available[fitting.index]]
    if len(training) <= dimensions:
        raise ValueError("Insufficient fitting examples for field PCA")
    pca = PCA(n_components=dimensions, svd_solver="randomized", random_state=SEED)
    pca.fit(field_vectors[training])
    joblib.dump(pca, directory / f"{field}_pca.joblib", compress=3)
    result = []
    for part in parts:
        values = pca.transform(field_vectors[part.index]).astype(np.float32)
        values[~available[part.index]] = 0
        result.append(pd.DataFrame(values, index=part.index, columns=[f"{field}_pca_{i}" for i in range(dimensions)]))
    return result


def train_fold(number, payload, table, semantic, visual, media_columns, history_manifest, manifest, stage):
    directory = OUT / f"fold_{number}"
    directory.mkdir(exist_ok=True)
    path = directory / "report.json"
    plan = json.loads((ROOT / "results" / "maria" / f"fold_{number}.json").read_text())
    plan = {key:plan[key] for key in ("fitting_authors", "calibration_authors", "validation_authors")}
    config = dict(version=1, parameters=PARAMETERS, features_sha256=manifest["features_sha256"],
                  field_vectors_sha256=manifest["field_vectors_sha256"], history_sha256=history_manifest["features_sha256"],
                  gate_strengths=[.01, .1, 1.], base_blend_iterations=800)
    report = json.loads(path.read_text()) if path.exists() else dict(fold=number, config=config, author_plan=plan,
        models={}, complete=False, scope="Unchanged outer author CV on official train; official test untouched")
    if report["config"] != config or report["author_plan"] != plan:
        raise ValueError("Saved research protocol changed")
    names = FEATURE_NAMES+["balanced_loss", "ordinal"] if stage == "content" else NAMES[8:] if stage == "fusion" else NAMES
    remaining = [name for name in names if name not in report["models"]]
    if not remaining:
        print(f"Fold {number}: {stage} already complete", flush=True)
        return
    fitting, calibration, validation = parts = split_by_plan(table, plan)
    predictions = pd.read_csv(directory / "predictions.csv") if (directory / "predictions.csv").exists() else validation[["author", "post", "label"]].reset_index(drop=True)
    if not predictions[["author", "post", "label"]].equals(validation[["author", "post", "label"]].reset_index(drop=True)):
        raise ValueError("Saved validation rows differ")
    for part, file in ((calibration, "calibration_keys.csv"), (validation, "predictions.csv")):
        if not part[["author", "post", "label"]].reset_index(drop=True).equals(pd.read_csv(HISTORY / f"fold_{number}" / file)[["author", "post", "label"]]):
            raise ValueError("v2 author alignment changed")
    calibration[["author", "post", "label"]].to_csv(directory / "calibration_keys.csv", index=False)
    numeric = payload["numeric"]+payload["extra_numeric"]
    cats = CATEGORY+["writing_system"]
    columns = numeric+cats+history_manifest["novelty_columns"]+history_manifest["neighbor_columns"]+media_columns
    if set(columns)&{"author", "post", "label", "class", "split", "reactions", "resid", "age_days", "log_followers"}:
        raise ValueError("Forbidden model feature")
    sem_pca = joblib.load(HISTORY / f"fold_{number}" / "semantic_pca.joblib")
    vis_pca = joblib.load(HISTORY / f"fold_{number}" / "visual_pca.joblib")
    joblib.dump(sem_pca, directory / "semantic_pca.joblib", compress=3)
    joblib.dump(vis_pca, directory / "visual_pca.joblib", compress=3)
    base = [pd.concat([part[columns], pd.DataFrame(sem_pca.transform(semantic[part.index]), index=part.index,
                   columns=[f"semantic_pca_{i}" for i in range(64)]),
                   pd.DataFrame(image_projection(vis_pca, visual[part.index], part.visual_available), index=part.index,
                   columns=[f"visual_pca_{i}" for i in range(64)])], axis=1) for part in parts]
    probabilities = {}
    field_vectors = np.load(OUT / "field_vectors.npz")
    sample = np.linspace(0, len(validation)-1, 64).astype(int)
    for name in remaining:
        if name in ("constant_blend", "adaptive_blend"):
            continue
        started = time.monotonic()
        matrices = [part.copy() for part in base]
        if name in FEATURE_NAMES:
            matrices = [pd.concat([matrix, part[manifest["groups"][name]]], axis=1) for matrix, part in zip(matrices, parts)]
        if name in ("separate_fields", "additional_fields"):
            fields = POLICY["fields"][:3] if name == "separate_fields" else ["additional_text"]
            for field in fields:
                projection = add_field_projection(parts, field_vectors[field], fitting, directory, field,
                    POLICY["quote_pca"] if name == "separate_fields" else POLICY["additional_pca"])
                matrices = [pd.concat([matrix, proj], axis=1) for matrix, proj in zip(matrices, projection)]
        if name == "blend_text":
            drop = media_columns+[f"visual_pca_{i}" for i in range(64)]
            matrices = [matrix.drop(columns=drop) for matrix in matrices]
        settings = dict(PARAMETERS)
        fixed = name.startswith("blend_")
        details = dict(features=len(matrices[0].columns), fixed_budget_without_calibration_labels=fixed,
                       script_sha256_at_fit=fingerprint(Path(__file__)))
        print(f"Fold {number}: fitting {name} ({details['features']} features)", flush=True)
        if name == "balanced_loss":
            counts = np.bincount(fitting.label, minlength=3)
            weights = np.sqrt(len(fitting)/(3*counts))
            settings["class_weights"] = weights.tolist()
            details["class_weights"] = weights.tolist()
        if name == "ordinal":
            settings["loss_function"] = "Logloss"
            cal_parts, val_parts, trees = [], [], []
            for cutoff in (0, 1):
                model = CatBoostClassifier(**settings)
                model.fit(matrices[0], (fitting.label>cutoff).astype(int), cat_features=cats,
                    eval_set=(matrices[1], (calibration.label>cutoff).astype(int)), early_stopping_rounds=100, use_best_model=True)
                model.save_model(str(directory / f"ordinal_gt{cutoff}.cbm"))
                reloaded = CatBoostClassifier().load_model(str(directory / f"ordinal_gt{cutoff}.cbm"))
                val_part = model.predict_proba(matrices[2])[:, 1]
                np.testing.assert_allclose(reloaded.predict_proba(matrices[2].iloc[sample])[:, 1], val_part[sample], atol=1e-8)
                cal_parts.append(model.predict_proba(matrices[1])[:, 1])
                val_parts.append(val_part)
                trees.append(int(model.tree_count_))
            cal, val = ordinal_probabilities(*cal_parts), ordinal_probabilities(*val_parts)
            details.update(trees=trees, crossings_calibration=int((cal_parts[1]>cal_parts[0]).sum()),
                           crossings_validation=int((val_parts[1]>val_parts[0]).sum()),
                           method="Two binary CatBoost thresholds with monotone projection; not CORAL")
        else:
            model = CatBoostClassifier(**settings)
            fit_args = {} if fixed else dict(eval_set=(matrices[1], calibration.label), early_stopping_rounds=100, use_best_model=True)
            model.fit(matrices[0], fitting.label, cat_features=cats, **fit_args)
            model.save_model(str(directory / f"{name}.cbm"))
            cal, val = ordered_probabilities(model, matrices[1]), ordered_probabilities(model, matrices[2])
            reloaded = CatBoostClassifier().load_model(str(directory / f"{name}.cbm"))
            np.testing.assert_allclose(reloaded.predict_proba(matrices[2].iloc[sample]), val[sample], atol=1e-8)
            details.update(trees=int(model.tree_count_), top_features=[dict(feature=k, importance=float(v))
                for k, v in sorted(zip(matrices[0].columns, model.feature_importances_), key=lambda pair: -pair[1])[:20]])
        details["roundtrip_passed"] = True
        save_json(directory / f"{name}_schema.json", dict(columns=matrices[0].columns.tolist(), categorical=cats,
            policy=POLICY, class_order=[0, 1, 2], cutoffs=payload["cutoffs"], parameters=settings,
            features_sha256=manifest["features_sha256"], field_vectors_sha256=manifest["field_vectors_sha256"]))
        record_predictions(name, cal, val, calibration, validation, payload["cutoffs"][0], report, predictions,
                           probabilities, directory, started, details)
    if stage in ("all", "fusion"):
        fit_mask = split_gate_authors(calibration)
        selection = ~fit_mask
        for name in ("blend_text", "blend_visual"):
            stored = np.load(directory / f"{name}_probabilities.npz")
            probabilities[name] = (stored["calibration"], stored["validation"])
        first, second = probabilities["blend_text"], probabilities["blend_visual"]
        for name in ("constant_blend", "adaptive_blend"):
            if name in report["models"]:
                continue
            started = time.monotonic()
            best = None
            if name == "constant_blend":
                for weight in (0., .25, .5, .75, 1.):
                    cal = (1-weight)*first[0]+weight*second[0]
                    decision = select_decision(calibration.label.to_numpy()[selection], cal[selection])
                    if best is None or decision["calibration_macro_f1"] > best["calibration_macro_f1"]:
                        best = {**decision, "visual_weight":weight}
                weight = best["visual_weight"]
                cal, val = [(1-weight)*a+weight*b for a, b in zip(first, second)]
                details = dict(weight_grid=[0., .25, .5, .75, 1.])
            else:
                raw = blend_features(calibration)
                mean, scale = raw[fit_mask].mean(axis=0), np.maximum(raw[fit_mask].std(axis=0), 1e-6)
                x = gate_matrix(calibration, mean, scale)
                xv = gate_matrix(validation, mean, scale)
                trials = []
                for strength in config["gate_strengths"]:
                    solution = minimize(gate_objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                        args=(x[fit_mask], first[0][fit_mask], second[0][fit_mask], calibration.label.to_numpy()[fit_mask], strength),
                        options=dict(maxiter=300, ftol=1e-10))
                    if not solution.success:
                        raise ValueError(f"Gate optimization failed: {solution.message}")
                    weight = expit(x @ solution.x)[:, None]
                    cal = (1-weight)*first[0]+weight*second[0]
                    decision = select_decision(calibration.label.to_numpy()[selection], cal[selection])
                    trials.append(dict(strength=strength, macro_f1=decision["calibration_macro_f1"], iterations=int(solution.nit)))
                    if best is None or decision["calibration_macro_f1"] > best["calibration_macro_f1"]:
                        best = {**decision, "strength":strength}
                        best_coef = solution.x.copy()
                weights = [expit(z @ best_coef)[:, None] for z in (x, xv)]
                cal, val = [(1-w)*a+w*b for w, a, b in zip(weights, first, second)]
                checkpoint = dict(mean=mean, scale=scale, coef=best_coef)
                joblib.dump(checkpoint, directory / "adaptive_blend.joblib")
                reload = joblib.load(directory / "adaptive_blend.joblib")
                np.testing.assert_allclose(expit(gate_matrix(validation, reload["mean"], reload["scale"]) @ reload["coef"]), weights[1][:, 0])
                details = dict(trials=trials, roundtrip_passed=True, mean_visual_weight=float(weights[1].mean()),
                    mean_visual_weight_empty=float(weights[1][validation.original_chars.eq(0)].mean()),
                    mean_visual_weight_with_images=float(weights[1][validation.visual_available.gt(0)].mean()))
            details.update(gate_fitting_authors=sorted(calibration.author[fit_mask].unique().tolist()),
                gate_selection_authors=sorted(calibration.author[selection].unique().tolist()),
                protocol="Base models fixed 800 trees on fitting authors only. Gate fitted on half calibration authors; regularization, constant weight and class multipliers chosen on other half. Outer authors untouched.")
            record_predictions(name, cal, val, calibration, validation, payload["cutoffs"][0], report, predictions,
                               probabilities, directory, started, details, best)
    report["complete"] = all(name in report["models"] for name in NAMES)
    save_json(path, report)


def summarize(table):
    reports, parts = [], []
    for number in (1, 2, 3):
        directory = OUT / f"fold_{number}"
        if not (directory / "report.json").exists():
            return
        reports.append(json.loads((directory / "report.json").read_text()))
        parts.append(pd.read_csv(directory / "predictions.csv").assign(fold=number))
    names = [name for name in NAMES if all(name in report["models"] for report in reports)]
    if not names:
        return
    combined = pd.concat([part[["author", "post", "label", "fold"]+names] for part in parts], ignore_index=True)
    coverage = combined.merge(table[["author", "post", "label"]], on=["author", "post", "label"], how="outer", validate="one_to_one", indicator=True)
    if not coverage._merge.eq("both").all():
        raise ValueError("Incomplete OOF coverage")
    reference = pd.read_csv(HISTORY / "oof_predictions.csv")[["author", "post", "label", "visual_neighbors"]].rename(columns={"visual_neighbors":"v2"})
    combined = combined.merge(reference, on=["author", "post", "label"], validate="one_to_one")
    comparisons = [(name, "v2") for name in names]
    if "adaptive_blend" in names:
        comparisons.append(("adaptive_blend", "constant_blend"))
    slices = combined.merge(table[["author", "post", "original_chars", "hist_n", "visual_available", "has_reshared"]], on=["author", "post"], validate="one_to_one")
    masks = dict(empty_original=slices.original_chars.eq(0), own_text=slices.original_chars.gt(0),
                 short_history=slices.hist_n.lt(10), images=slices.visual_available.gt(0), no_images=slices.visual_available.eq(0), quotes=slices.has_reshared.eq(1))
    result = dict(scope="Development OOF within official train, no new official test or all-train fit", rows=len(combined),
        authors=int(combined.author.nunique()), models_completed=names, complete=all(report["complete"] for report in reports),
        oof={name:metrics(combined.label, combined[name]) for name in ["v2"]+names},
        folds={str(r["fold"]):{name:r["models"][name]["metrics"]["macro_f1"] for name in names} for r in reports},
        comparisons_95={name+"_minus_"+ref:bootstrap_authors(combined, name, ref) for name, ref in comparisons},
        slices={key:dict(n=int(mask.sum()), macro_f1={name:metrics(slices.loc[mask,"label"], slices.loc[mask,name])["macro_f1"] for name in ["v2"]+names}) for key,mask in masks.items()},
        uncertainty="500 paired author bootstrap resamples conditional on predictions. Does not include repeated model selection or seed uncertainty.")
    combined.to_csv(OUT / "oof_predictions.csv", index=False)
    save_json(OUT / "comparison.json", result)
    print(json.dumps(dict(oof={k:v["macro_f1"] for k,v in result["oof"].items()}, comparisons_95=result["comparisons_95"]), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["content", "fusion", "all"], default="all")
    parser.add_argument("--fold", type=int, choices=[1,2,3])
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    payload, table, own, expanded, _ = load_inputs()
    table, history_manifest = load_history(table)
    table, visual, media_columns, _ = load_visual(table)
    manifest = json.loads((OUT / "features_manifest.json").read_text())
    if manifest["features_sha256"] != fingerprint(OUT / "features.joblib") or manifest["field_vectors_sha256"] != fingerprint(OUT / "field_vectors.npz"):
        raise ValueError("Feature cache hash differs")
    features = joblib.load(OUT / "features.joblib")
    if not features[["author", "post"]].equals(table[["author", "post"]]):
        raise ValueError("Feature alignment differs")
    table = pd.concat([table, features.drop(columns=["author", "post"])], axis=1)
    with threadpool_limits(limits=6):
        if not args.summarize_only:
            for number in ([args.fold] if args.fold else [1,2,3]):
                train_fold(number, payload, table, np.concatenate([own, expanded], axis=1), visual,
                           media_columns, history_manifest, manifest, args.stage)
        summarize(table)


if __name__ == "__main__":
    main()
