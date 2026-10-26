"""Controlled repairs for empty-original posts, preserving other predictions."""
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
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from train_maria import SEED,CATEGORY,metrics,bootstrap_authors
from scripts.semantic_models import load_inputs
from scripts.history_models import load_history,PARAMETERS
from scripts.visual_models import load_visual,image_projection
from scripts.lecture_models import split_by_plan,ordered_probabilities
from scripts.pilot_models import restrict_support
from scripts.calibrate_lecture_decisions import select_decision,GRID
from scripts.semantic_embeddings import save_json,fingerprint
from scripts.ocr_empty_posts import OUT,OCR

BASE_NAMES = ["old_visual_route","group_calibration","guarded_history","empty_specialist"]
OCR_NAMES = ["ocr_metadata","ocr_semantic"]
ALL_NAMES = BASE_NAMES+OCR_NAMES+["selected_route","balanced_route"]
SPECIALIST = dict(iterations=500,depth=4,learning_rate=.05,loss_function="MultiClass",l2_leaf_reg=10,
                  random_seed=SEED,thread_count=6,verbose=False,allow_writing_files=False)


def route_predictions(base,empty_predictions,mask):
    result = np.asarray(base).copy()
    mask = np.asarray(mask,dtype=bool)
    replacement = np.asarray(empty_predictions)
    if len(mask)!=len(result) or mask.sum()!=len(replacement):
        raise ValueError("Invalid route alignment")
    result[mask] = replacement
    return result


def guard_context(table,columns):
    result = table.copy()
    preserve = {"current_available","past_available","past_missing","coherence"}
    masked = [c for c in columns if c.startswith("nbr_expanded_") or
              (c.startswith("nov_expanded_") and c.removeprefix("nov_expanded_") not in preserve)]
    result.loc[result.original_chars.eq(0),masked] = np.nan
    return result,masked


def reference(number,folder,model,calibration,validation):
    directory = ROOT / "results" / folder / f"fold_{number}"
    for table,filename in ((calibration,"calibration_keys.csv"),(validation,"predictions.csv")):
        expected = pd.read_csv(directory / filename)[["author","post","label"]]
        if not table[["author","post","label"]].reset_index(drop=True).equals(expected):
            raise ValueError("Reference author fold differs")
    report = json.loads((directory / "report.json").read_text())
    with np.load(directory / f"{model}_probabilities.npz") as saved:
        c,v = saved["calibration"],saved["validation"]
    config = report["models"][model]["calibration"]
    return c,v,config


def choose_route(labels,empty_mask,table):
    baseline_empty = f1_score(np.asarray(labels)[empty_mask],table.current.to_numpy()[empty_mask],average="macro",labels=[0,1,2],zero_division=0)
    chosen,best = "current",f1_score(labels,table.current,average="macro",labels=[0,1,2])
    choices = []
    for name in BASE_NAMES+OCR_NAMES:
        empty = f1_score(np.asarray(labels)[empty_mask],table[name].to_numpy()[empty_mask],average="macro",labels=[0,1,2],zero_division=0)
        overall = f1_score(labels,table[name],average="macro",labels=[0,1,2])
        eligible = empty>=baseline_empty-1e-12
        choices.append(dict(model=name,overall=float(overall),empty=float(empty),eligible=bool(eligible)))
        if eligible and overall>best+1e-12:
            chosen,best = name,float(overall)
    return dict(chosen=chosen,calibration_overall=float(best),baseline_empty=float(baseline_empty),candidates=choices,
        selection="Maximize inner overall macro-F1 subject to no inner empty-group decline; outer labels unused")


def balanced_decision(labels,mask,base_predictions,probabilities,base_config):
    labels,mask = np.asarray(labels,dtype=int),np.asarray(mask,dtype=bool)
    def matrix(y,p):
        return np.bincount(3*y+p,minlength=9).reshape(3,3)
    def score(cm):
        denominator = cm.sum(0)+cm.sum(1)
        return float(np.divide(2*cm.diagonal(),denominator,out=np.zeros(3),where=denominator!=0).mean())
    outside = matrix(labels[~mask],base_predictions[~mask])
    baseline = matrix(labels[mask],base_predictions[mask])
    minimum = score(baseline)
    best = dict(chosen="current",multipliers=base_config["multipliers"],
                calibration_overall=score(outside+baseline),calibration_empty=minimum)
    for name,probability in probabilities.items():
        for below in GRID:
            for above in GRID:
                weights = [below,1.,above]
                inside = matrix(labels[mask],(probability*weights).argmax(1))
                empty,overall = score(inside),score(outside+inside)
                if empty>=minimum-1e-12 and overall>best["calibration_overall"]+1e-12:
                    best = dict(chosen=name,multipliers=weights,calibration_overall=overall,calibration_empty=empty)
    best.update(minimum_empty=minimum,selection="Joint model and multiplier selection on inner overall F1, constrained by inner empty F1",grid=list(GRID))
    return best


def train_fold(number,stage,payload,table,semantic,visual,media_columns,history_manifest,ocr_vectors,ocr_columns):
    directory = OUT / f"fold_{number}"
    directory.mkdir(exist_ok=True)
    path = directory / "report.json"
    config = dict(version=1,history_sha256=history_manifest["features_sha256"],global_parameters=PARAMETERS,
                  specialist_parameters=SPECIALIST,scope="Empty original only; preserve current predictions otherwise")
    plan = json.loads((ROOT / "results" / "maria" / f"fold_{number}.json").read_text())
    parts = fitting,calibration,validation = split_by_plan(table,plan)
    masks = [part.original_chars.eq(0).to_numpy() for part in parts]
    fm,cm,vm = masks
    report = json.loads(path.read_text()) if path.exists() else dict(fold=number,config=config,models={},complete=False,
        fitting_empty=int(fm.sum()),calibration_empty=int(cm.sum()),validation_empty=int(vm.sum()),
        author_plan={key:plan[key] for key in ("fitting_authors","calibration_authors","validation_authors")})
    if report["config"]!=config or any(report["author_plan"][k]!=plan[k] for k in report["author_plan"]):
        raise ValueError("Protocol differs from saved models")
    if stage=="ocr":
        ocr_hash = fingerprint(OCR / "posts.joblib")
        if report.get("ocr_posts_sha256",ocr_hash)!=ocr_hash:
            raise ValueError("Saved specialist models use a different OCR corpus")
        report["ocr_posts_sha256"] = ocr_hash
    current_c,current_v,current_config = reference(number,"semantic_history","visual_neighbors",calibration,validation)
    base_c,base_v = ((prob*current_config["multipliers"]).argmax(1) for prob in (current_c,current_v))
    ctable = pd.read_csv(directory / "calibration_predictions.csv") if (directory / "calibration_predictions.csv").exists() else calibration[["author","post","label"]].reset_index(drop=True).assign(current=base_c)
    vtable = pd.read_csv(directory / "predictions.csv") if (directory / "predictions.csv").exists() else validation[["author","post","label"]].reset_index(drop=True).assign(current=base_v)
    for frame,part in ((ctable,calibration),(vtable,validation)):
        if not frame[["author","post","label"]].equals(part[["author","post","label"]].reset_index(drop=True)):
            raise ValueError("Saved candidate row alignment differs")

    def save_candidate(name,c,v,details,decision=None,started=None):
        c = restrict_support(c,calibration.loc[cm],payload["cutoffs"][0])
        v = restrict_support(v,validation.loc[vm],payload["cutoffs"][0])
        decision = decision or select_decision(calibration.label.to_numpy()[cm],c)
        cp,vp = ((prob*decision["multipliers"]).argmax(1) for prob in (c,v))
        ctable[name],vtable[name] = route_predictions(base_c,cp,cm),route_predictions(base_v,vp,vm)
        full_c,full_v = current_c.copy(),current_v.copy()
        full_c[cm],full_v[vm] = c,v
        np.savez_compressed(directory / f"{name}_probabilities.npz",calibration=full_c,validation=full_v)
        report["models"][name] = dict(metrics=metrics(validation.label,vtable[name]),empty=metrics(validation.label.to_numpy()[vm],vp),
            calibration_empty=metrics(calibration.label.to_numpy()[cm],cp),calibration=decision,details=details,
            seconds=round(time.monotonic()-started,1) if started else 0.)
        ctable.to_csv(directory / "calibration_predictions.csv",index=False)
        vtable.to_csv(directory / "predictions.csv",index=False)
        save_json(path,report)
        print(f"Fold {number} {name}: all {report['models'][name]['metrics']['macro_f1']:.4f}; empty {report['models'][name]['empty']['macro_f1']:.4f}",flush=True)

    if stage=="base":
        if "old_visual_route" not in report["models"]:
            c,v,decision = reference(number,"visual","multimodal_catboost",calibration,validation)
            save_candidate("old_visual_route",c[cm],v[vm],dict(trained=False,source="Previous visual model"),decision)
        if "group_calibration" not in report["models"]:
            save_candidate("group_calibration",current_c[cm],current_v[vm],dict(trained=False,source="Current model, empty-specific class multipliers"))
    requested = ["guarded_history","empty_specialist"] if stage=="base" else OCR_NAMES
    requested = [name for name in requested if name not in report["models"]]
    if requested:
        numeric = payload["numeric"]+payload["extra_numeric"]
        cats = CATEGORY+["writing_system"]
        spca = joblib.load(ROOT / "results" / "semantic" / f"fold_{number}" / "semantic_pca.joblib")
        vpca = joblib.load(ROOT / "results" / "visual" / f"fold_{number}" / "visual_pca.joblib")
        joblib.dump(spca,directory / "semantic_pca.joblib",compress=3)
        joblib.dump(vpca,directory / "visual_pca.joblib",compress=3)
        projection = [pd.concat([pd.DataFrame(spca.transform(semantic[part.index]),index=part.index,columns=[f"semantic_pca_{i}" for i in range(64)]),
            pd.DataFrame(image_projection(vpca,visual[part.index],part.visual_available),index=part.index,columns=[f"visual_pca_{i}" for i in range(64)])],axis=1) for part in parts]
        history_columns = history_manifest["novelty_columns"]+history_manifest["neighbor_columns"]
        for name in requested:
            started = time.monotonic()
            guarded = name=="guarded_history"
            columns = numeric+cats+media_columns+(history_columns if guarded else [])
            masked = []
            if guarded:
                source,masked = guard_context(table,history_columns)
                matrices = [pd.concat([source.loc[part.index,columns],projection[i]],axis=1) for i,part in enumerate(parts)]
            else:
                matrices = [pd.concat([part[columns],projection[i]],axis=1).loc[mask].copy() for i,(part,mask) in enumerate(zip(parts,masks))]
            if name in OCR_NAMES:
                matrices = [pd.concat([matrix,part.loc[mask,ocr_columns]],axis=1) for matrix,part,mask in zip(matrices,parts,masks)]
            if name=="ocr_semantic":
                fit_indices = fitting.index[fm & fitting.ocr_available.to_numpy(dtype=bool)]
                if len(fit_indices)<33:
                    raise ValueError("Insufficient OCR fitting examples for the fixed PCA")
                opca = PCA(n_components=32,svd_solver="randomized",random_state=SEED).fit(ocr_vectors[fit_indices])
                joblib.dump(opca,directory / "ocr_pca.joblib",compress=3)
                for i,(part,mask) in enumerate(zip(parts,masks)):
                    subset = part.loc[mask]
                    values = image_projection(opca,ocr_vectors[subset.index],subset.ocr_available)
                    matrices[i] = pd.concat([matrices[i],pd.DataFrame(values,index=subset.index,columns=[f"ocr_pca_{j}" for j in range(32)])],axis=1)
            labels = [part.label if guarded else part.label.to_numpy()[mask] for part,mask in zip(parts,masks)]
            model = CatBoostClassifier(**(PARAMETERS if guarded else SPECIALIST))
            print(f"Fold {number}: training {name} on {len(matrices[0])} posts",flush=True)
            model.fit(matrices[0],labels[0],cat_features=cats,eval_set=(matrices[1],labels[1]),
                      early_stopping_rounds=100 if guarded else 60,use_best_model=True)
            model.save_model(str(directory / f"{name}.cbm"))
            c,v = ordered_probabilities(model,matrices[1]),ordered_probabilities(model,matrices[2])
            reloaded = CatBoostClassifier().load_model(str(directory / f"{name}.cbm"))
            chosen = np.linspace(0,len(v)-1,min(64,len(v))).astype(int)
            np.testing.assert_allclose(reloaded.predict_proba(matrices[2].iloc[chosen]),v[chosen],atol=1e-8)
            save_json(directory / f"{name}_schema.json",dict(columns=matrices[0].columns.tolist(),categorical=cats,
                masked_history_columns=masked,cutoffs=payload["cutoffs"],class_order=[0,1,2]))
            details = dict(trained=True,roundtrip_passed=True,trees=int(model.tree_count_),fitting_posts=len(matrices[0]),
                masked_history_columns=masked,parameters=PARAMETERS if guarded else SPECIALIST)
            save_candidate(name,c[cm] if guarded else c,v[vm] if guarded else v,details,started=started)
    if stage=="ocr" and all(name in report["models"] for name in BASE_NAMES+OCR_NAMES):
        selection = choose_route(calibration.label,cm,ctable)
        chosen = selection["chosen"]
        ctable["selected_route"],vtable["selected_route"] = ctable[chosen],vtable[chosen]
        report["models"]["selected_route"] = dict(metrics=metrics(validation.label,vtable.selected_route),
            empty=metrics(validation.label.to_numpy()[vm],vtable.selected_route.to_numpy()[vm]),details=selection)
        candidates = {"current":(current_c[cm],current_v[vm])}
        # group_calibration has the same raw probabilities as current, so skip it here.
        for candidate in ["old_visual_route","guarded_history","empty_specialist"]+OCR_NAMES:
            with np.load(directory / f"{candidate}_probabilities.npz") as saved:
                candidates[candidate] = (saved["calibration"][cm],saved["validation"][vm])
        balanced = balanced_decision(calibration.label,cm,base_c,{name:value[0] for name,value in candidates.items()},current_config)
        cp,vp = ((value*balanced["multipliers"]).argmax(1) for value in candidates[balanced["chosen"]])
        ctable["balanced_route"],vtable["balanced_route"] = route_predictions(base_c,cp,cm),route_predictions(base_v,vp,vm)
        report["models"]["balanced_route"] = dict(metrics=metrics(validation.label,vtable.balanced_route),
            empty=metrics(validation.label.to_numpy()[vm],vp),details=balanced)
        ctable.to_csv(directory / "calibration_predictions.csv",index=False)
        vtable.to_csv(directory / "predictions.csv",index=False)
        print(f"Fold {number}: selected {chosen} using inner authors",flush=True)
        print(f"Fold {number}: balanced route {balanced['chosen']}, all {report['models']['balanced_route']['metrics']['macro_f1']:.4f}, empty {report['models']['balanced_route']['empty']['macro_f1']:.4f}",flush=True)
    report["base_complete"] = all(name in report["models"] for name in BASE_NAMES)
    report["complete"] = all(name in report["models"] for name in ALL_NAMES)
    save_json(path,report)


def summarize(table):
    reports,parts = [],[]
    for number in (1,2,3):
        directory = OUT / f"fold_{number}"
        if not (directory / "report.json").exists():
            return
        report = json.loads((directory / "report.json").read_text())
        if not report.get("base_complete"):
            return
        reports.append(report)
        parts.append(pd.read_csv(directory / "predictions.csv").assign(fold=number))
    names = [name for name in ALL_NAMES if all(name in report["models"] for report in reports)]
    combined = pd.concat([p[["author","post","label","fold","current"]+names] for p in parts],ignore_index=True)
    combined = combined.merge(table[["author","post","original_chars","has_reshared","visual_available"]],on=["author","post"],how="outer",validate="one_to_one",indicator=True)
    if not combined._merge.eq("both").all():
        raise ValueError("OOF coverage differs")
    combined = combined.drop(columns="_merge")
    empty = combined.original_chars.eq(0)
    if any(not np.array_equal(combined.loc[~empty,name],combined.loc[~empty,"current"]) for name in names):
        raise ValueError("A candidate changed nonempty predictions")
    masks = dict(empty=empty,has_original=~empty,empty_with_images=empty&combined.visual_available.eq(1),
        empty_without_images=empty&combined.visual_available.eq(0),empty_reshared=empty&combined.has_reshared.eq(1),empty_not_reshared=empty&combined.has_reshared.eq(0))
    report = dict(n=len(combined),authors=int(combined.author.nunique()),models_completed=names,
        scope="Development CV on official train; same nested author folds; official test unused",
        oof={name:metrics(combined.label,combined[name]) for name in ["current"]+names},
        slices={key:dict(n=int(mask.sum()),models={name:metrics(combined.loc[mask,"label"],combined.loc[mask,name]) for name in ["current"]+names}) for key,mask in masks.items()},
        folds={str(r["fold"]):{name:dict(overall=r["models"][name]["metrics"]["macro_f1"],empty=r["models"][name]["empty"]["macro_f1"]) for name in names} for r in reports},
        nonempty_predictions_unchanged=True,
        all_saved_classifiers_roundtrip_passed=all(v["details"].get("roundtrip_passed") for r in reports for v in r["models"].values() if v["details"].get("trained")))
    comparisons = [("old_visual_route","current"),("group_calibration","current"),("guarded_history","group_calibration"),("empty_specialist","group_calibration")]
    if "selected_route" in names:
        comparisons += [("ocr_semantic","ocr_metadata"),("ocr_semantic","empty_specialist"),("selected_route","current"),("balanced_route","current")]
    report["comparisons_95"] = {a+"_minus_"+b:dict(overall=bootstrap_authors(combined,a,b),empty=bootstrap_authors(combined[empty],a,b)) for a,b in comparisons}
    report["uncertainty"] = "Paired author bootstrap, conditional on predictions; no selection or random-seed uncertainty"
    combined.to_csv(OUT / "oof_predictions.csv",index=False)
    save_json(OUT / "comparison.json",report)
    print(json.dumps({name:dict(overall=report["oof"][name]["macro_f1"],empty=report["slices"]["empty"]["models"][name]["macro_f1"]) for name in ["current"]+names},indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["base","ocr"],default="base")
    parser.add_argument("--fold",type=int,choices=[1,2,3])
    parser.add_argument("--summarize-only",action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    payload,table,own,expanded,_ = load_inputs()
    table,history_manifest = load_history(table)
    table,visual,media_columns,_ = load_visual(table)
    ocr_vectors,ocr_columns = None,[]
    if args.stage=="ocr":
        manifest = json.loads((OCR / "embedding_manifest.json").read_text())
        ocr = joblib.load(OCR / "posts.joblib")
        if manifest["posts_sha256"]!=fingerprint(OCR / "posts.joblib") or not table[["author","post"]].equals(ocr[["author","post"]]):
            raise ValueError("OCR cache mismatch")
        ocr_columns = [c for c in ocr if c.startswith("ocr_") and c!="ocr_text"]
        table = pd.concat([table,ocr[ocr_columns]],axis=1)
        ocr_vectors = np.load(OCR / "vectors.npy",mmap_mode="r")
    semantic = np.concatenate([own,expanded],axis=1)
    with threadpool_limits(limits=6):
        if not args.summarize_only:
            for number in ([args.fold] if args.fold else [1,2,3]):
                train_fold(number,args.stage,payload,table,semantic,visual,media_columns,history_manifest,ocr_vectors,ocr_columns)
        summarize(table)


if __name__ == "__main__":
    main()
