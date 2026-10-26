"""Three-seed checks, ordinal/quote interaction, and inner-author decision stability."""
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
from threadpoolctl import threadpool_limits

from train_maria import SEED, CATEGORY, metrics
from scripts.semantic_models import load_inputs
from scripts.lecture_models import split_by_plan, ordered_probabilities
from scripts.history_models import load_history, PARAMETERS
from scripts.visual_models import load_visual, image_projection
from scripts.priority_models import ordinal_probabilities
from scripts.priority_features import OUT as PRIORITY, HISTORY
from scripts.semantic_embeddings import save_json, fingerprint
from scripts.calibrate_lecture_decisions import GRID, select_decision
from scripts.pilot_models import restrict_support

OUT = ROOT / "results" / "stability_research"
FAMILIES = ["quote", "ordinal", "joint"]
SEEDS = [SEED, SEED+1, SEED+2]
RULES = ["grid", "fine_grid", "bootstrap_grid", "bootstrap_fine"]
FINE_GRID = sorted(set(GRID) | {float(np.sqrt(a*b)) for a,b in zip(GRID[:-1], GRID[1:])})
INDIVIDUALS = [f"{family}_s{i}" for family in FAMILIES for i in range(3)]
MEANS = [f"{family}_mean_{rule}" for family in FAMILIES for rule in RULES]
NAMES = INDIVIDUALS+MEANS
POLICY = dict(version=1, families=FAMILIES, seeds=SEEDS, parameters=PARAMETERS,
    ensemble="Equal arithmetic mean of three support-restricted probability matrices; no weight fitting",
    rules=RULES, coarse_grid=list(GRID), fine_grid=FINE_GRID, decision_bootstraps=100,
    bootstrap_seed=SEED+900, aggregation="Geometric mean of the class multipliers selected in author bootstrap resamples",
    stopping="Existing inner calibration authors, patience 100; no outer/test labels used for stopping or decisions",
    primary="joint_mean_grid versus ordinal_mean_grid; other comparisons are exploratory")


def macro_from_confusion(matrices):
    matrices = np.asarray(matrices)
    denominator = matrices.sum(axis=-2)+matrices.sum(axis=-1)
    return np.divide(2*np.diagonal(matrices,axis1=-2,axis2=-1), denominator,
                     out=np.zeros_like(denominator,dtype=float), where=denominator!=0).mean(axis=-1)


def author_confusions(labels, predicted, authors):
    labels, predicted = np.asarray(labels,dtype=int), np.asarray(predicted,dtype=int)
    if labels.ndim!=1 or labels.shape!=predicted.shape or len(labels)!=len(authors):
        raise ValueError("Invalid author/label alignment")
    if np.any((labels<0)|(labels>2)|(predicted<0)|(predicted>2)):
        raise ValueError("Expected three classes")
    unique, codes = np.unique(np.asarray(authors),return_inverse=True)
    return np.bincount(codes*9+labels*3+predicted,minlength=len(unique)*9).reshape(-1,3,3)


def bootstrap_difference(labels, predicted, reference, authors, repeats=500):
    first = author_confusions(labels,predicted,authors)
    second = author_confusions(labels,reference,authors)
    rng, values = np.random.default_rng(SEED), []
    for offset in range(0,repeats,64):
        chosen = rng.integers(0,len(first),size=(min(64,repeats-offset),len(first)))
        values.extend((macro_from_confusion(first[chosen].sum(axis=1))-macro_from_confusion(second[chosen].sum(axis=1))).tolist())
    return dict(low=float(np.quantile(values,.025)),high=float(np.quantile(values,.975)),resamples=repeats,unit="author")


def stable_decisions(labels, probabilities, authors, grid=GRID, repeats=100):
    labels, probabilities = np.asarray(labels,dtype=int),np.asarray(probabilities,dtype=float)
    if not len(labels) or probabilities.shape!=(len(labels),3) or len(authors)!=len(labels):
        raise ValueError("Invalid calibration alignment")
    if not np.isfinite(probabilities).all() or (probabilities<0).any() or not np.allclose(probabilities.sum(axis=1),1):
        raise ValueError("Invalid calibration probabilities")
    if not grid or np.any(np.asarray(grid)<=0) or repeats<1:
        raise ValueError("Invalid decision configuration")
    choices = np.array([[below,1.,above] for below in grid for above in grid])
    matrices = np.stack([author_confusions(labels,(probabilities*choice).argmax(axis=1),authors) for choice in choices])
    scores = macro_from_confusion(matrices.sum(axis=1))
    best = int(scores.argmax())
    rng = np.random.default_rng(POLICY["bootstrap_seed"])
    n_authors = matrices.shape[1]
    draws = rng.integers(0,n_authors,size=(repeats,n_authors))
    counts = np.stack([np.bincount(row,minlength=n_authors) for row in draws])
    weighted = np.tensordot(counts.astype(float),matrices.astype(float),axes=(1,1))
    chosen = macro_from_confusion(weighted).argmax(axis=1)
    multipliers = np.exp(np.log(choices[chosen]).mean(axis=0))
    predicted = (probabilities*multipliers).argmax(axis=1)
    bag_score = float(macro_from_confusion(author_confusions(labels,predicted,authors).sum(axis=0)))
    common, frequencies = np.unique(chosen,return_counts=True)
    order = np.argsort(-frequencies,kind="stable")[:5]
    return dict(calibration_macro_f1=float(scores[best]),multipliers=choices[best].tolist()), dict(
        calibration_macro_f1=bag_score,multipliers=multipliers.tolist(),resamples=repeats,unit="calibration author",
        multiplier_quantiles={str(q):np.quantile(choices[chosen],q,axis=0).tolist() for q in (.1,.5,.9)},
        frequent_rules=[dict(multipliers=choices[common[i]].tolist(),count=int(frequencies[i])) for i in order])


def load_all():
    payload, table, own, expanded, _ = load_inputs()
    table, history_manifest = load_history(table)
    table, visual, media_columns, _ = load_visual(table)
    manifest = json.loads((PRIORITY / "features_manifest.json").read_text())
    features = joblib.load(PRIORITY / "features.joblib")
    if manifest["features_sha256"] != fingerprint(PRIORITY / "features.joblib") or not features[["author","post"]].equals(table[["author","post"]]):
        raise ValueError("Priority feature cache differs")
    quote_columns = manifest["groups"]["history_quote"]
    table = pd.concat([table,features[quote_columns]],axis=1)
    config = dict(policy=POLICY,history_sha256=history_manifest["features_sha256"],
                  priority_features_sha256=manifest["features_sha256"],quote_columns=quote_columns)
    return payload,table,np.concatenate([own,expanded],axis=1),visual,media_columns,history_manifest,config


def train_fold(number,payload,table,semantic,visual,media_columns,history_manifest,config):
    directory = OUT / f"fold_{number}"
    directory.mkdir(exist_ok=True)
    path = directory / "report.json"
    plan = json.loads((ROOT / "results" / "maria" / f"fold_{number}.json").read_text())
    plan = {key:plan[key] for key in ("fitting_authors","calibration_authors","validation_authors")}
    report = json.loads(path.read_text()) if path.exists() else dict(fold=number,config=config,author_plan=plan,
        models={},complete=False,scope="Development CV on unchanged official-train author splits; official test unused")
    if report["config"]!=config or report["author_plan"]!=plan:
        raise ValueError("Saved stability experiment changed")
    if report["complete"]:
        print(f"Fold {number} already complete",flush=True)
        return
    fitting,calibration,validation = parts = split_by_plan(table,plan)
    for part,file in ((calibration,"calibration_keys.csv"),(validation,"predictions.csv")):
        if not part[["author","post","label"]].reset_index(drop=True).equals(pd.read_csv(PRIORITY / f"fold_{number}" / file)[["author","post","label"]]):
            raise ValueError("Reference prediction alignment changed")
    cal_keys, val_keys = [part[["author","post","label"]].reset_index(drop=True) for part in (calibration,validation)]
    cal_keys.to_csv(directory / "calibration_keys.csv",index=False)
    predictions = pd.read_csv(directory / "predictions.csv") if (directory / "predictions.csv").exists() else val_keys.copy()
    if not predictions[["author","post","label"]].equals(val_keys):
        raise ValueError("Saved prediction rows changed")
    cats = CATEGORY+["writing_system"]
    columns = payload["numeric"]+payload["extra_numeric"]+cats+history_manifest["novelty_columns"]+history_manifest["neighbor_columns"]+media_columns
    sem = joblib.load(HISTORY / f"fold_{number}" / "semantic_pca.joblib")
    img = joblib.load(HISTORY / f"fold_{number}" / "visual_pca.joblib")
    joblib.dump(sem,directory / "semantic_pca.joblib",compress=3)
    joblib.dump(img,directory / "visual_pca.joblib",compress=3)
    matrices = [pd.concat([part[columns],pd.DataFrame(sem.transform(semantic[part.index]),index=part.index,
        columns=[f"semantic_pca_{i}" for i in range(64)]),pd.DataFrame(image_projection(img,visual[part.index],part.visual_available),
        index=part.index,columns=[f"visual_pca_{i}" for i in range(64)])],axis=1) for part in parts]
    quote_matrices = [pd.concat([matrix,part[config["quote_columns"]]],axis=1) for matrix,part in zip(matrices,parts)]
    sample = np.linspace(0,len(validation)-1,64).astype(int)
    low = payload["cutoffs"][0]

    def record(name,cal,val,decision,details,started):
        cal,val = restrict_support(cal,calibration,low),restrict_support(val,validation,low)
        predictions[name] = (val*decision["multipliers"]).argmax(axis=1)
        np.savez_compressed(directory / f"{name}_probabilities.npz",calibration=cal,validation=val)
        report["models"][name] = dict(metrics=metrics(validation.label,predictions[name]),calibration=decision,
            details=details,seconds=round(time.monotonic()-started,1))
        predictions.to_csv(directory / "predictions.csv",index=False)
        save_json(path,report)
        print(f"Fold {number} {name}: macro-F1 {report['models'][name]['metrics']['macro_f1']:.4f}",flush=True)

    for family in FAMILIES:
        x = matrices if family=="ordinal" else quote_matrices
        if set(x[0].columns)&{"author","post","label","split","reactions","resid","age_days","log_followers"}:
            raise ValueError("Forbidden model features")
        for index,seed in enumerate(SEEDS):
            name = f"{family}_s{index}"
            if name in report["models"]:
                continue
            started = time.monotonic()
            settings = {**PARAMETERS,"random_seed":seed}
            details = dict(family=family,seed=seed,features=len(x[0].columns),script_sha256_at_fit=fingerprint(Path(__file__)))
            if family in ("quote","ordinal") and index==0:
                source_name = "history_quote" if family=="quote" else "ordinal"
                source = PRIORITY / f"fold_{number}"
                schema = json.loads((source / f"{source_name}_schema.json").read_text())
                expected = {**settings,"loss_function":"Logloss"} if family=="ordinal" else settings
                if schema["columns"]!=x[0].columns.tolist() or schema["parameters"]!=expected:
                    raise ValueError("Reused seed-zero implementation differs")
                stored = np.load(source / f"{source_name}_probabilities.npz")
                cal,val = stored["calibration"],stored["validation"]
                old_report = json.loads((source / "report.json").read_text())["models"][source_name]
                files = [source / f"{source_name}_probabilities.npz", source / f"{source_name}_schema.json"]
                files += [source / f"ordinal_gt{k}.cbm" for k in (0,1)] if family=="ordinal" else [source / "history_quote.cbm"]
                details.update(reused=True,roundtrip_passed=old_report["details"]["roundtrip_passed"],
                    trees=old_report["details"]["trees"],sources={f.relative_to(ROOT).as_posix():fingerprint(f) for f in files})
                decision = select_decision(calibration.label,cal)
                if decision!=old_report["calibration"]:
                    raise ValueError("Reused decision no longer matches")
            else:
                print(f"Fold {number}: fitting {name} ({len(x[0].columns)} features)",flush=True)
                settings["loss_function"] = "MultiClass" if family=="quote" else "Logloss"
                heads = [None] if family=="quote" else [0,1]
                cal_parts,val_parts,trees = [],[],[]
                for head in heads:
                    target = fitting.label if head is None else (fitting.label>head).astype(int)
                    inner_target = calibration.label if head is None else (calibration.label>head).astype(int)
                    model = CatBoostClassifier(**settings)
                    model.fit(x[0],target,cat_features=cats,eval_set=(x[1],inner_target),early_stopping_rounds=100,use_best_model=True)
                    filename = f"{name}.cbm" if head is None else f"{name}_gt{head}.cbm"
                    model.save_model(str(directory / filename))
                    reloaded = CatBoostClassifier().load_model(str(directory / filename))
                    pcal,pval = model.predict_proba(x[1]),model.predict_proba(x[2])
                    np.testing.assert_allclose(reloaded.predict_proba(x[2].iloc[sample]),pval[sample],atol=1e-8)
                    if list(model.classes_)!=([0,1,2] if head is None else [0,1]):
                        raise ValueError("Unexpected class order")
                    cal_parts.append(pcal if head is None else pcal[:,1])
                    val_parts.append(pval if head is None else pval[:,1])
                    trees.append(int(model.tree_count_))
                cal = cal_parts[0] if family=="quote" else ordinal_probabilities(*cal_parts)
                val = val_parts[0] if family=="quote" else ordinal_probabilities(*val_parts)
                cal,val = restrict_support(cal,calibration,low),restrict_support(val,validation,low)
                decision = select_decision(calibration.label,cal)
                details.update(reused=False,roundtrip_passed=True,trees=trees)
                save_json(directory / f"{name}_schema.json",dict(columns=x[0].columns.tolist(),categorical=cats,parameters=settings,
                    class_order=[0,1,2],cutoffs=payload["cutoffs"],config=config,heads=heads))
            record(name,cal,val,decision,details,started)
        # Equal weights are fixed beforehand. Calibrating the mean is separate from calibrating each seed.
        probabilities = [np.load(directory / f"{family}_s{i}_probabilities.npz") for i in range(3)]
        cal = np.mean([p["calibration"] for p in probabilities],axis=0)
        val = np.mean([p["validation"] for p in probabilities],axis=0)
        cal,val = restrict_support(cal,calibration,low),restrict_support(val,validation,low)
        if not all(f"{family}_mean_{rule}" in report["models"] for rule in RULES):
            started = time.monotonic()
            coarse,bag_coarse = stable_decisions(calibration.label,cal,calibration.author,GRID)
            fine,bag_fine = stable_decisions(calibration.label,cal,calibration.author,FINE_GRID)
            for rule,decision in zip(RULES,(coarse,fine,bag_coarse,bag_fine)):
                name = f"{family}_mean_{rule}"
                if name not in report["models"]:
                    record(name,cal,val,decision,dict(seeds=SEEDS,weights=[1/3]*3,rule=rule),started)
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
        raise ValueError("Incomplete OOF coverage")
    refs = pd.read_csv(PRIORITY / "oof_predictions.csv")[["author","post","label","v2","ordinal","history_quote"]].rename(columns={"ordinal":"old_ordinal","history_quote":"old_quote"})
    combined = combined.merge(refs,on=["author","post","label"],validate="one_to_one")
    for name,reference in (("quote_s0","old_quote"),("ordinal_s0","old_ordinal")):
        np.testing.assert_array_equal(combined[name],combined[reference])
    comparisons = [(name,"old_ordinal") for name in MEANS]
    comparisons += [(f"{family}_mean_grid",f"{family}_s0") for family in FAMILIES]
    comparisons += [("joint_mean_grid","ordinal_mean_grid"),("joint_mean_grid","quote_mean_grid")]
    comparisons += [(f"{family}_mean_{rule}",f"{family}_mean_grid") for family in FAMILIES for rule in RULES[1:]]
    masks_table = combined.merge(table[["author","post","original_chars","hist_n","has_reshared","visual_available"]],on=["author","post"],validate="one_to_one")
    masks = dict(empty_original=masks_table.original_chars.eq(0),short_history=masks_table.hist_n.lt(10),
                 quotes=masks_table.has_reshared.eq(1),no_images=masks_table.visual_available.eq(0))
    names = ["v2","old_ordinal"]+NAMES
    scores = {name:metrics(combined.label,combined[name]) for name in names}
    result = dict(complete=True,rows=len(combined),authors=int(combined.author.nunique()),config=reports[0]["config"],
        scope="Development OOF within official train; three seeds on same author folds; no new official test/all-train fit",
        oof=scores,folds={str(r["fold"]):{name:r["models"][name]["metrics"]["macro_f1"] for name in NAMES} for r in reports},
        seed_spread={family:dict(scores=[scores[f"{family}_s{i}"]["macro_f1"] for i in range(3)],
            mean=float(np.mean([scores[f"{family}_s{i}"]["by_class"]["macro avg"]["f1-score"] for i in range(3)])),
            std=float(np.std([scores[f"{family}_s{i}"]["by_class"]["macro avg"]["f1-score"] for i in range(3)],ddof=1))) for family in FAMILIES},
        comparisons_95={name+"_minus_"+ref:bootstrap_difference(combined.label,combined[name],combined[ref],combined.author) for name,ref in comparisons},
        slices={key:dict(n=int(mask.sum()),macro_f1={name:metrics(masks_table.loc[mask,"label"],masks_table.loc[mask,name])["macro_f1"] for name in names}) for key,mask in masks.items()},
        uncertainty="500 paired author resamples conditional on predictions. Three seeds measure only a small part of optimization variance. Repeated model selection and multiple comparisons are not corrected.",
        classifier_counts=dict(new=36,reused=9))
    combined.to_csv(OUT / "oof_predictions.csv",index=False)
    save_json(OUT / "comparison.json",result)
    print(json.dumps(dict(oof={name:scores[name]["macro_f1"] for name in ["v2","old_ordinal"]+MEANS},seed_spread=result["seed_spread"]),indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold",type=int,choices=[1,2,3])
    parser.add_argument("--summarize-only",action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    data = load_all()
    config = data[-1]
    path = OUT / "protocol.json"
    if path.exists() and json.loads(path.read_text())!=config:
        raise ValueError("Experiment protocol changed")
    save_json(path,config)
    with threadpool_limits(limits=6):
        if not args.summarize_only:
            for number in ([args.fold] if args.fold else [1,2,3]):
                train_fold(number,*data)
        summarize(data[1])


if __name__ == "__main__":
    main()
