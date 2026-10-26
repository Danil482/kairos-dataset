"""Text-first, then visual ablations of strictly earlier semantic author history."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

from train_maria import SEED, CATEGORY, metrics, bootstrap_authors
from scripts.semantic_models import load_inputs, record_predictions
from scripts.lecture_models import split_by_plan, ordered_probabilities
from scripts.semantic_embeddings import save_json, fingerprint
from scripts.semantic_history import OUT, POLICY
from scripts.visual_models import load_visual, image_projection

NAMES = ["text_novelty", "text_neighbors", "visual_novelty", "visual_neighbors"]
PARAMETERS = dict(iterations=800, depth=6, learning_rate=.045, loss_function="MultiClass", l2_leaf_reg=5,
                  random_seed=SEED, thread_count=6, verbose=False, allow_writing_files=False)


def load_history(table):
    manifest = json.loads((OUT / "features_manifest.json").read_text())
    features = joblib.load(OUT / "features.joblib")
    if not manifest["complete"] or manifest["config"]["policy"] != POLICY or manifest["features_sha256"] != fingerprint(OUT / "features.joblib"):
        raise ValueError("History features are incomplete or incompatible")
    if not table[["author","post"]].equals(features[["author","post"]]):
        raise ValueError("History feature rows do not match model inputs")
    return pd.concat([table,features.drop(columns=["author","post"])],axis=1),manifest


def train_fold(number, names, payload, table, semantic, visual, media_columns, manifest):
    directory = OUT / f"fold_{number}"
    directory.mkdir(exist_ok=True)
    path = directory / "report.json"
    config = dict(features_sha256=manifest["features_sha256"], parameters=PARAMETERS, policy=POLICY, version=1)
    plan = json.loads((ROOT / "results" / "maria" / f"fold_{number}.json").read_text())
    report = json.loads(path.read_text()) if path.exists() else dict(fold=number, config=config,models={},complete=False,
        scope="Development CV on official train, unchanged nested author folds",
        author_plan={key:plan[key] for key in ("fitting_authors","calibration_authors","validation_authors")})
    if report["config"] != config or any(report["author_plan"][key] != plan[key] for key in report["author_plan"]):
        raise ValueError("Saved history model protocol changed")
    remaining = [name for name in names if name not in report["models"]]
    if not remaining:
        print(f"Fold {number}: requested models already complete",flush=True)
        return
    fitting,calibration,validation = parts = split_by_plan(table,plan)
    reference = ROOT / "results" / "semantic" / f"fold_{number}"
    for part,filename in ((calibration,"calibration_keys.csv"),(validation,"predictions.csv")):
        if not part[["author","post","label"]].reset_index(drop=True).equals(pd.read_csv(reference / filename)[["author","post","label"]]):
            raise ValueError("Author fold alignment changed")
    predictions = pd.read_csv(directory / "predictions.csv") if (directory / "predictions.csv").exists() else validation[["author","post","label"]].reset_index(drop=True)
    if not predictions[["author","post","label"]].equals(validation[["author","post","label"]].reset_index(drop=True)):
        raise ValueError("Previously saved prediction order changed")
    calibration[["author","post","label"]].to_csv(directory / "calibration_keys.csv",index=False)
    numeric = payload["numeric"]+payload["extra_numeric"]
    cats = CATEGORY+["writing_system"]
    pca = joblib.load(reference / "semantic_pca.joblib")
    joblib.dump(pca,directory / "semantic_pca.joblib",compress=3)
    semantic_columns = [f"semantic_pca_{i}" for i in range(64)]
    projections = [pd.DataFrame(pca.transform(semantic[part.index]),index=part.index,columns=semantic_columns) for part in parts]
    image_columns, image_projections = [], None
    if any(name.startswith("visual_") for name in remaining):
        image_pca = joblib.load(ROOT / "results" / "visual" / f"fold_{number}" / "visual_pca.joblib")
        joblib.dump(image_pca,directory / "visual_pca.joblib",compress=3)
        image_columns = [f"visual_pca_{i}" for i in range(64)]
        image_projections = [pd.DataFrame(image_projection(image_pca,visual[part.index],part.visual_available),index=part.index,columns=image_columns) for part in parts]
    for name in remaining:
        started = time.monotonic()
        with_images = name.startswith("visual_")
        columns = numeric+cats+manifest["novelty_columns"]
        if name.endswith("neighbors"):
            columns += manifest["neighbor_columns"]
        if with_images:
            columns += media_columns
        if {"author","post","label","class","split","reactions","resid","age_days","log_followers"}.intersection(columns):
            raise ValueError("Invalid model feature allowlist")
        matrices = [pd.concat([part[columns],projections[i]]+([image_projections[i]] if with_images else []),axis=1) for i,part in enumerate(parts)]
        print(f"Fold {number}: fitting {name}, {len(matrices[0].columns)} features",flush=True)
        model = CatBoostClassifier(**PARAMETERS)
        model.fit(matrices[0],fitting.label,cat_features=cats,eval_set=(matrices[1],calibration.label),
                  early_stopping_rounds=100,use_best_model=True)
        model.save_model(str(directory / f"{name}.cbm"))
        cal,val = ordered_probabilities(model,matrices[1]),ordered_probabilities(model,matrices[2])
        reloaded = CatBoostClassifier().load_model(str(directory / f"{name}.cbm"))
        # Spread checks across the validation authors, not just one author's first posts.
        selected = np.linspace(0,len(validation)-1,64).astype(int)
        np.testing.assert_allclose(reloaded.predict_proba(matrices[2].iloc[selected]),val[selected],atol=1e-8)
        importance = sorted(zip(matrices[0].columns,model.feature_importances_),key=lambda pair:-pair[1])
        groups = dict(novelty=set(manifest["novelty_columns"]),neighbors=set(manifest["neighbor_columns"]),
                      semantic=set(semantic_columns),visual=set(image_columns),media_metadata=set(media_columns),other=set(numeric+cats))
        save_json(directory / f"{name}_schema.json",dict(columns=matrices[0].columns.tolist(),categorical=cats,
            class_order=[0,1,2],cutoffs=payload["cutoffs"],history_config=manifest["config"],
            semantic_order=["own","expanded"],visual_order=["mean","first_manifest"] if with_images else [],
            features_sha256=manifest["features_sha256"]))
        details = dict(trees=int(model.tree_count_),roundtrip_passed=True,
            feature_importance_by_group={group:float(sum(value for key,value in importance if key in members)) for group,members in groups.items()},
            top_features=[dict(feature=key,importance=float(value)) for key,value in importance[:25]],
            script_sha256_at_fit=fingerprint(Path(__file__)))
        record_predictions(name,cal,val,calibration,validation,payload["cutoffs"][0],report,predictions,{},directory,started,details)
    report["text_complete"] = all(name in report["models"] for name in NAMES[:2])
    report["complete"] = all(name in report["models"] for name in NAMES)
    save_json(path,report)


def summarize(table):
    reports,parts = [],[]
    for number in (1,2,3):
        directory = OUT / f"fold_{number}"
        if not (directory / "report.json").exists():
            return
        report = json.loads((directory / "report.json").read_text())
        if not report.get("text_complete"):
            return
        reports.append(report)
        parts.append(pd.read_csv(directory / "predictions.csv").assign(fold=number))
    names = [name for name in NAMES if all(name in report["models"] for report in reports)]
    combined = pd.concat([part[["author","post","label","fold"]+names] for part in parts],ignore_index=True)
    coverage = combined.merge(table[["author","post","label"]],on=["author","post","label"],how="outer",indicator=True,validate="one_to_one")
    if not coverage._merge.eq("both").all():
        raise ValueError("Incomplete OOF coverage")
    for folder,refs in (("semantic",["e5_history_catboost","semantic_fusion"]),("visual",["multimodal_catboost"])):
        baseline = pd.read_csv(ROOT / "results" / folder / "oof_predictions.csv")
        combined = combined.merge(baseline[["author","post","label"]+refs],on=["author","post","label"],how="left",validate="one_to_one")
    if combined.isna().any().any():
        raise ValueError("Reference prediction alignment failed")
    refs = ["e5_history_catboost","semantic_fusion","multimodal_catboost"]
    comparisons = [(name,"multimodal_catboost" if name.startswith("visual") else "e5_history_catboost") for name in names]
    comparisons += [("text_neighbors","text_novelty"),("text_neighbors","multimodal_catboost")]
    if "visual_neighbors" in names:
        comparisons += [("visual_neighbors","visual_novelty"),("visual_neighbors","text_neighbors")]
    report = dict(n=len(combined),authors=int(combined.author.nunique()),models_completed=names,
        scope="Development CV on official train; no new official test evaluation",
        oof={name:metrics(combined.label,combined[name]) for name in refs+names},
        folds={str(r["fold"]):{name:r["models"][name]["metrics"]["macro_f1"] for name in names} for r in reports},
        comparisons_95={name+"_minus_"+reference:bootstrap_authors(combined,name,reference) for name,reference in comparisons},
        uncertainty="500 paired author bootstrap resamples conditional on predictions; no model-selection uncertainty",
        all_saved_classifiers_roundtrip_passed=all(r["models"][name]["details"]["roundtrip_passed"] for r in reports for name in names))
    slices = combined.merge(table[["author","post","original_chars","hist_n","nov_own_novelty_100","nov_expanded_past_available"]],on=["author","post"],validate="one_to_one")
    threshold = float(slices.nov_own_novelty_100.quantile(.75))
    masks = dict(empty_original=slices.original_chars.eq(0),has_original=slices.original_chars.gt(0),
        short_history=slices.hist_n.lt(10),high_semantic_novelty=slices.nov_own_novelty_100.ge(threshold))
    report["slices"] = {key:dict(n=int(mask.sum()),models={name:metrics(slices.loc[mask,"label"],slices.loc[mask,name]) for name in refs+names}) for key,mask in masks.items()}
    report["descriptive_novelty_75_percentile"] = threshold
    report["best_development_model"] = max(refs+names,key=lambda name:report["oof"][name]["by_class"]["macro avg"]["f1-score"])
    combined.to_csv(OUT / "oof_predictions.csv",index=False)
    save_json(OUT / "comparison.json",report)
    print(json.dumps(dict(oof_macro_f1={name:value["macro_f1"] for name,value in report["oof"].items()},comparisons_95=report["comparisons_95"]),indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["text","visual","all"],default="text")
    parser.add_argument("--fold",type=int,choices=[1,2,3])
    parser.add_argument("--summarize-only",action="store_true")
    args = parser.parse_args()
    payload,table,own,expanded,_ = load_inputs()
    table,manifest = load_history(table)
    visual,media_columns = None,[]
    if args.stage in ("visual","all"):
        table,visual,media_columns,_ = load_visual(table)
    names = NAMES[:2] if args.stage=="text" else NAMES[2:] if args.stage=="visual" else NAMES
    with threadpool_limits(limits=6):
        if not args.summarize_only:
            for number in ([args.fold] if args.fold else [1,2,3]):
                train_fold(number,names,payload,table,np.concatenate([own,expanded],axis=1),visual,media_columns,manifest)
        summarize(table)


if __name__ == "__main__":
    main()
