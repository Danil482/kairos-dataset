"""Disjoint stopping/decision authors, separate quote/document, and nested E5 capacity."""
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
from threadpoolctl import threadpool_limits

from train_maria import SEED,CATEGORY,metrics
from scripts.stability_models import load_all,SEEDS,FINE_GRID,stable_decisions,bootstrap_difference,OUT as PREVIOUS
from scripts.history_models import PARAMETERS
from scripts.priority_features import OUT as PRIORITY,HISTORY
from scripts.priority_models import ordinal_probabilities
from scripts.lecture_models import split_by_plan
from scripts.visual_models import image_projection
from scripts.semantic_embeddings import save_json,fingerprint
from scripts.calibrate_lecture_decisions import select_decision
from scripts.pilot_models import restrict_support

OUT = ROOT / "results" / "separated_research"
VARIANTS = ["base64","fields64","e5_128","e5_256"]
RULES = ["grid","bootstrap_fine"]
NAMES = [f"{variant}_s0_{rule}" for variant in VARIANTS for rule in RULES]+[f"{variant}_mean_{rule}" for variant in ("baseline","selected") for rule in RULES]
POLICY = dict(version=1,seeds=SEEDS,parameters=PARAMETERS,variants=VARIANTS,split_seed=SEED+1700,
    stopping_fraction=.5,decision_rule="100 author bootstrap choices on dense multiplier grid, geometric mean",
    selection="Maximum decision-author bootstrap-rule macro-F1 of seed zero; ties keep earlier variant; no outer-label selection",
    ensembles="Three seeds for base64 and selected variant only, equal probability weights; other variants screened with seed zero",
    pca="Preserve old fitting-only 64 components; PCA of their reconstruction residual adds 64 or 192 nested directions",
    separate_fields=["quote_text","document_text"],field_dimensions=16,
    primary="selected_mean_bootstrap_fine minus baseline_mean_bootstrap_fine")


def split_stopping_decision(calibration):
    authors = np.array(sorted(calibration.author.unique()))
    if len(authors)<2:
        raise ValueError("Need at least two inner authors")
    np.random.default_rng(POLICY["split_seed"]).shuffle(authors)
    stopping = set(authors[:len(authors)//2])
    mask = calibration.author.isin(stopping)
    return calibration.loc[mask].copy(),calibration.loc[~mask].copy()


def residual_vectors(pca,vectors):
    return np.asarray(vectors)-pca.inverse_transform(pca.transform(vectors))


def extend_projection(original,residual,dimensions):
    if dimensions not in (64,128,256) or original.shape[1]!=64 or len(original)!=len(residual):
        raise ValueError("Invalid nested semantic projection")
    return np.concatenate([original,residual[:,:dimensions-64]],axis=1)


def choose_variant(scores):
    if set(scores)!=set(VARIANTS) or not np.isfinite(list(scores.values())).all():
        raise ValueError("Incomplete/non-finite inner screening scores")
    return max(VARIANTS,key=lambda variant:scores[variant])


def train_fold(number,data,config):
    payload,table,semantic,visual,media_columns,history_manifest,previous_config = data
    directory = OUT / f"fold_{number}"
    directory.mkdir(exist_ok=True)
    path = directory / "report.json"
    old_plan = json.loads((ROOT / "results" / "maria" / f"fold_{number}.json").read_text())
    fitting,calibration,validation = split_by_plan(table,old_plan)
    stopping,decision = split_stopping_decision(calibration)
    parts = [fitting,stopping,decision,validation]
    plan = {name:sorted(part.author.unique().tolist()) for name,part in zip(("fitting","stopping","decision","validation"),parts)}
    report = json.loads(path.read_text()) if path.exists() else dict(fold=number,config=config,author_plan=plan,
        fits={},models={},complete=False,scope="Unchanged outer train author folds; disjoint fitting/stopping/decision/validation authors; no official test")
    if report["config"]!=config or report["author_plan"]!=plan:
        raise ValueError("Saved protocol or author roles differ")
    if report["complete"]:
        print(f"Fold {number} already complete",flush=True)
        return
    if any(set(plan[a])&set(plan[b]) for a in plan for b in plan if a!=b):
        raise ValueError("Overlapping author roles")
    val_keys = validation[["author","post","label"]].reset_index(drop=True)
    if not val_keys.equals(pd.read_csv(PREVIOUS / f"fold_{number}" / "predictions.csv")[["author","post","label"]]):
        raise ValueError("Outer validation changed")
    predictions = pd.read_csv(directory / "predictions.csv") if (directory / "predictions.csv").exists() else val_keys.copy()
    if not predictions[["author","post","label"]].equals(val_keys):
        raise ValueError("Saved prediction order differs")
    for name,part in zip(("fitting","stopping","decision","validation"),parts):
        part[["author","post","label"]].to_csv(directory / f"{name}_keys.csv",index=False)
    save_json(path,report)
    cats = CATEGORY+["writing_system"]
    columns = payload["numeric"]+payload["extra_numeric"]+cats+history_manifest["novelty_columns"]+history_manifest["neighbor_columns"]+media_columns+previous_config["quote_columns"]
    if set(columns)&{"author","post","label","split","reactions","resid","age_days","log_followers"}:
        raise ValueError("Forbidden feature")
    old_pca = joblib.load(HISTORY / f"fold_{number}" / "semantic_pca.joblib")
    image_pca = joblib.load(HISTORY / f"fold_{number}" / "visual_pca.joblib")
    joblib.dump(old_pca,directory / "semantic_pca64.joblib",compress=3)
    joblib.dump(image_pca,directory / "visual_pca.joblib",compress=3)
    residual_path = directory / "semantic_residual_pca192.joblib"
    if residual_path.exists():
        residual_pca = joblib.load(residual_path)
    else:
        print(f"Fold {number}: fitting residual E5 PCA on fitting authors only",flush=True)
        residual_pca = PCA(n_components=192,svd_solver="randomized",random_state=SEED)
        residual_pca.fit(residual_vectors(old_pca,semantic[fitting.index]))
        joblib.dump(residual_pca,residual_path,compress=3)
    original_projection = [old_pca.transform(semantic[part.index]) for part in parts]
    extra_projection = [residual_pca.transform(residual_vectors(old_pca,semantic[part.index])) for part in parts]
    total_variance = float(np.var(semantic[fitting.index],axis=0,ddof=1,dtype=np.float64).sum())
    old_variance = float(old_pca.explained_variance_.sum())
    overlap = float(np.abs(old_pca.components_ @ residual_pca.components_.T).max())
    if overlap>1e-3:
        raise ValueError("Additional PCA directions are not orthogonal to the old projection")
    report["pca"] = dict(fitting_posts=len(fitting),max_basis_overlap=overlap,
        explained_variance={str(n):(old_variance+float(residual_pca.explained_variance_[:n-64].sum()))/total_variance for n in (64,128,256)},
        hashes={name:fingerprint(directory/name) for name in ("semantic_pca64.joblib","semantic_residual_pca192.joblib","visual_pca.joblib")})
    shared = [pd.concat([part[columns],pd.DataFrame(image_projection(image_pca,visual[part.index],part.visual_available),
        index=part.index,columns=[f"visual_pca_{i}" for i in range(64)])],axis=1) for part in parts]
    # Preserve the old feature order: tabular/history/media, semantic PCA, image PCA, quote history.
    old_schema = json.loads((PREVIOUS / f"fold_{number}" / "joint_s0_schema.json").read_text())["columns"]
    matrices = {}
    for variant,dimensions in (("base64",64),("e5_128",128),("e5_256",256)):
        matrices[variant] = []
        for i,part in enumerate(parts):
            values = extend_projection(original_projection[i],extra_projection[i],dimensions)
            matrix = pd.concat([shared[i],pd.DataFrame(values,index=part.index,columns=[f"semantic_pca_{j}" for j in range(dimensions)])],axis=1)
            order = old_schema+[f"semantic_pca_{j}" for j in range(64,dimensions)]
            matrices[variant].append(matrix[order])
    fields = np.load(PRIORITY / "field_vectors.npz")
    field_features = joblib.load(PRIORITY / "features.joblib")
    if not field_features[["author","post"]].equals(table[["author","post"]]):
        raise ValueError("Separate-field metadata alignment differs")
    matrices["fields64"] = [matrix.copy() for matrix in matrices["base64"]]
    for field in POLICY["separate_fields"]:
        field_pca = joblib.load(PRIORITY / f"fold_{number}" / f"{field}_pca.joblib")
        joblib.dump(field_pca,directory / f"{field}_pca.joblib",compress=3)
        metadata = [field+suffix for suffix in ("_available","_log_chars","_own_similarity")]
        vectors = fields[field]
        for i,part in enumerate(parts):
            projection = field_pca.transform(vectors[part.index]).astype(np.float32)
            projection[np.linalg.norm(vectors[part.index],axis=1)<1e-8] = 0
            matrices["fields64"][i] = pd.concat([matrices["fields64"][i],field_features.loc[part.index,metadata],
                pd.DataFrame(projection,index=part.index,columns=[f"{field}_pca_{j}" for j in range(16)])],axis=1)
    save_json(path,report)
    sample = np.linspace(0,len(validation)-1,64).astype(int)

    def fit(variant,index):
        name = f"{variant}_s{index}"
        if name in report["fits"]:
            saved = np.load(directory / f"{name}_probabilities.npz")
            return saved["decision"],saved["validation"]
        started = time.monotonic()
        x = matrices[variant]
        settings = {**PARAMETERS,"random_seed":SEEDS[index],"loss_function":"Logloss"}
        print(f"Fold {number}: fitting {name}, {len(x[0].columns)} features",flush=True)
        dp,vp,trees = [],[],[]
        for threshold in (0,1):
            model = CatBoostClassifier(**settings)
            model.fit(x[0],(fitting.label>threshold).astype(int),cat_features=cats,
                eval_set=(x[1],(stopping.label>threshold).astype(int)),early_stopping_rounds=100,use_best_model=True)
            file = directory / f"{name}_gt{threshold}.cbm"
            model.save_model(str(file))
            d,v = model.predict_proba(x[2])[:,1],model.predict_proba(x[3])[:,1]
            reloaded = CatBoostClassifier().load_model(str(file))
            np.testing.assert_allclose(reloaded.predict_proba(x[3].iloc[sample])[:,1],v[sample],atol=1e-8)
            dp.append(d); vp.append(v); trees.append(int(model.tree_count_))
        d = restrict_support(ordinal_probabilities(*dp),decision,payload["cutoffs"][0])
        v = restrict_support(ordinal_probabilities(*vp),validation,payload["cutoffs"][0])
        np.savez_compressed(directory / f"{name}_probabilities.npz",decision=d,validation=v)
        save_json(directory / f"{name}_schema.json",dict(columns=x[0].columns.tolist(),categorical=cats,settings=settings,
            variant=variant,seed=SEEDS[index],class_order=[0,1,2],cutoffs=payload["cutoffs"],config=config,
            stopping_authors=plan["stopping"],decision_authors=plan["decision"]))
        report["fits"][name] = dict(seed=SEEDS[index],variant=variant,trees=trees,roundtrip_passed=True,
            seconds=round(time.monotonic()-started,1),script_sha256=fingerprint(Path(__file__)))
        save_json(path,report)
        return d,v

    def record(prefix,d,v):
        if all(f"{prefix}_{rule}" in report["models"] for rule in RULES):
            return
        rules = {"grid":select_decision(decision.label,d),
                 "bootstrap_fine":stable_decisions(decision.label,d,decision.author,FINE_GRID)[1]}
        np.savez_compressed(directory / f"{prefix}_decision_probabilities.npz",decision=d,validation=v)
        for rule,selected in rules.items():
            name = f"{prefix}_{rule}"
            predictions[name] = (v*selected["multipliers"]).argmax(axis=1)
            report["models"][name] = dict(decision=selected,metrics=metrics(validation.label,predictions[name]))
            print(f"Fold {number} {name}: macro-F1 {report['models'][name]['metrics']['macro_f1']:.4f}",flush=True)
        predictions.to_csv(directory / "predictions.csv",index=False)
        save_json(path,report)

    for variant in VARIANTS:
        d,v = fit(variant,0)
        record(f"{variant}_s0",d,v)
    scores = {variant:report["models"][f"{variant}_s0_bootstrap_fine"]["decision"]["calibration_macro_f1"] for variant in VARIANTS}
    selected = choose_variant(scores)
    if "selected_variant" in report and report["selected_variant"]!=selected:
        raise ValueError("Saved inner selection changed")
    report["selected_variant"] = selected
    report["screening_scores"] = scores
    save_json(path,report)
    print(f"Fold {number}: inner decision authors selected {selected}",flush=True)
    for variant in dict.fromkeys(["base64",selected]):
        for index in (1,2):
            fit(variant,index)
    for prefix,variant in (("baseline", "base64"),("selected",selected)):
        arrays = [fit(variant,index) for index in range(3)]
        d = restrict_support(np.mean([a[0] for a in arrays],axis=0),decision,payload["cutoffs"][0])
        v = restrict_support(np.mean([a[1] for a in arrays],axis=0),validation,payload["cutoffs"][0])
        record(prefix+"_mean",d,v)
    report["complete"] = all(name in report["models"] for name in NAMES)
    save_json(path,report)


def summarize(table):
    reports,parts = [],[]
    for number in (1,2,3):
        directory = OUT / f"fold_{number}"
        if not (directory / "report.json").exists():
            return
        report = json.loads((directory / "report.json").read_text())
        if not report["complete"]:
            return
        reports.append(report)
        parts.append(pd.read_csv(directory / "predictions.csv").assign(fold=number))
    combined = pd.concat(parts,ignore_index=True)
    coverage = combined.merge(table[["author","post","label"]],on=["author","post","label"],how="outer",indicator=True,validate="one_to_one")
    if not coverage._merge.eq("both").all():
        raise ValueError("Incomplete external author coverage")
    reference = pd.read_csv(PREVIOUS / "oof_predictions.csv")[["author","post","label","joint_s0","joint_mean_bootstrap_fine"]].rename(columns={"joint_s0":"old_single","joint_mean_bootstrap_fine":"old_best"})
    combined = combined.merge(reference,on=["author","post","label"],validate="one_to_one")
    comparisons = [(name,"old_best") for name in NAMES if "_mean_" in name]
    comparisons += [(f"{variant}_s0_bootstrap_fine","base64_s0_bootstrap_fine") for variant in VARIANTS[1:]]
    comparisons += [("selected_mean_bootstrap_fine","baseline_mean_bootstrap_fine"),("base64_s0_bootstrap_fine","old_single")]
    slices = combined.merge(table[["author","post","original_chars","hist_n","has_reshared"]],on=["author","post"],validate="one_to_one")
    names = ["old_single","old_best"]+NAMES
    masks = dict(empty_original=slices.original_chars.eq(0),short_history=slices.hist_n.lt(10),quotes=slices.has_reshared.eq(1))
    result = dict(complete=True,rows=len(combined),authors=int(combined.author.nunique()),config=reports[0]["config"],
        scope="Development CV only; four disjoint author roles; official test untouched",
        oof={name:metrics(combined.label,combined[name]) for name in names},
        selected_variants={str(r["fold"]):r["selected_variant"] for r in reports},
        author_counts={str(r["fold"]):{role:len(authors) for role,authors in r["author_plan"].items()} for r in reports},
        pca={str(r["fold"]):r["pca"] for r in reports},
        classifiers_trained=2*sum(len(r["fits"]) for r in reports),
        folds={str(r["fold"]):{name:r["models"][name]["metrics"]["macro_f1"] for name in NAMES} for r in reports},
        comparisons_95={name+"_minus_"+ref:bootstrap_difference(combined.label,combined[name],combined[ref],combined.author) for name,ref in comparisons},
        slices={key:dict(n=int(mask.sum()),macro_f1={name:metrics(slices.loc[mask,"label"],slices.loc[mask,name])["macro_f1"] for name in names}) for key,mask in masks.items()},
        uncertainty="500 paired author bootstrap resamples conditional on fitted predictions; model-selection and multiple-comparison uncertainty not included")
    combined.to_csv(OUT / "oof_predictions.csv",index=False)
    save_json(OUT / "comparison.json",result)
    print(json.dumps(dict(oof={k:v["macro_f1"] for k,v in result["oof"].items()},selected=result["selected_variants"],classifiers=result["classifiers_trained"]),indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold",type=int,choices=[1,2,3])
    parser.add_argument("--summarize-only",action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    data = load_all()
    fields_manifest = json.loads((PRIORITY / "features_manifest.json").read_text())
    if fingerprint(PRIORITY / "field_vectors.npz")!=fields_manifest["field_vectors_sha256"]:
        raise ValueError("Field vectors changed")
    config = dict(policy=POLICY,inputs=data[-1],field_vectors_sha256=fields_manifest["field_vectors_sha256"])
    path = OUT / "protocol.json"
    if path.exists() and json.loads(path.read_text())!=config:
        raise ValueError("Experiment protocol changed")
    save_json(path,config)
    with threadpool_limits(limits=6):
        if not args.summarize_only:
            for number in ([args.fold] if args.fold else [1,2,3]):
                train_fold(number,data,config)
        summarize(data[1])


if __name__ == "__main__":
    main()
