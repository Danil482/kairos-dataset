"""Reproducible second-stage experiments; worker commands share immutable inputs.

All feature selection and tuning use the fixed author folds within released train.
The public experiment entrypoint uses a frozen selection and does not tune on test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from kairos.experiment import (ROOT, KEYS, FeatureContext, fingerprint, metrics,
                               numeric_model, save_json, digest)
from kairos.experiment_features import load_experiment_data
from kairos.experiment_cache import load_frozen_embeddings
from kairos.experiment_selection import parameter_grid, simplex_weights, passes_gate
from kairos.experiment_uncertainty import paired_author_bootstrap


def atomic_dump(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    joblib.dump(value, tmp, compress=0)
    tmp.replace(path)


def prepare(data_dir, output, config):
    output.mkdir(parents=True, exist_ok=True)
    table, history, groups, report = load_experiment_data(data_dir)
    table, history = table.reset_index(drop=True), history.reset_index(drop=True)
    try:
        dense, metadata = load_frozen_embeddings(history, data_dir / "cache/e5")
    except FileNotFoundError:
        from kairos.experiment_encode import encode_history
        settings = {k: v for k, v in config["encoder"].items() if k != "enabled"}
        dense, metadata = encode_history(history, data_dir / "cache/e5", settings)
    if table.loc[table.split.eq("test"), "label"].notna().any():
        raise ValueError("Test labels exposed to stage2")
    train_idx = np.flatnonzero(table.split.eq("train"))
    fold_authors = None
    if config.get("folds_file"):
        fold_authors = json.loads((ROOT / config["folds_file"]).read_text())["fold_authors"]
        if set(fold_authors) != {"0", "1", "2"}:
            raise ValueError("Exactly three explicit author folds are required")
        expected_authors = set(table.iloc[train_idx].author)
        seen = set()
        folds = []
        for fold in range(3):
            authors = fold_authors[str(fold)]
            selected_authors = set(authors)
            if (not authors or len(authors) != len(selected_authors)
                    or seen & selected_authors or not selected_authors <= expected_authors):
                raise ValueError("Explicit folds must contain disjoint unique train authors")
            seen.update(selected_authors)
            mask = table.iloc[train_idx].author.isin(selected_authors).to_numpy()
            folds.append((train_idx[~mask], train_idx[mask]))
        if seen != expected_authors:
            raise ValueError("Explicit folds do not cover every train author")
    else:
        folds = [(train_idx[a], train_idx[b]) for a, b in GroupKFold(3).split(
            train_idx, groups=table.iloc[train_idx].author)]
    identity = fingerprint(config, report)
    identity["encoder"] = metadata
    identity["svd_blas_threads"] = 1
    if fold_authors is not None:
        identity["fold_authors_sha256"] = digest(fold_authors)
    bundle = {"table": table, "history": history, "groups": groups, "report": report,
              "folds": folds, "dense_path": str((data_dir / "cache/e5/history.npy").resolve()),
              "config": config, "identity": identity, "run_id": digest(identity)}
    atomic_dump(bundle, output / "prepared.joblib")
    save_json(output / "run_manifest.json", identity)
    save_json(output / "data_report.json", report)
    print(f"Prepared {len(train_idx)} train rows, 3 author folds", flush=True)
    return bundle


def load_bundle(output):
    bundle = joblib.load(output / "prepared.joblib")
    current = fingerprint(bundle["config"], bundle["report"])
    previous = bundle["identity"]
    if current["source_sha256"] != previous["source_sha256"]:
        raise ValueError("Source changed after prepare; start a new experiment run")
    if current["versions"] != previous["versions"]:
        raise ValueError("Worker dependency versions differ from prepare")
    if bundle["config"].get("folds_file"):
        authors = json.loads((ROOT / bundle["config"]["folds_file"]).read_text())["fold_authors"]
        if digest(authors) != previous["fold_authors_sha256"]:
            raise ValueError("Explicit author folds changed after prepare")
    return bundle


def indices(bundle, fold):
    table = bundle["table"]
    if fold == "final":
        return (np.flatnonzero(table.split.eq("train")),
                np.flatnonzero(table.split.eq("test")))
    return bundle["folds"][int(fold)]


def load_artifact(output, name, bundle):
    value = joblib.load(output / name)
    if value["run_id"] != bundle["run_id"]:
        raise ValueError(f"Stale artifact: {name}")
    return value


def load_decision(output, name, bundle):
    value = json.loads((output / name).read_text())
    if value.get("run_id") != bundle["run_id"]:
        raise ValueError(f"Stale selection: {name}")
    return value


def feature_path(fold, block):
    return f"features/{fold}_{block}.joblib"


def stable_inner_folds(table, fit, n_splits):
    """Balance author counts with explicit, platform-independent tie breaks."""
    counts = table.iloc[fit].groupby("author", sort=False).size().to_dict()
    bins = [set() for _ in range(n_splits)]
    sizes = [0] * n_splits
    for author in sorted(counts, key=lambda author: (-counts[author], str(author))):
        target = min(range(n_splits), key=lambda fold: (sizes[fold], fold))
        bins[target].add(author)
        sizes[target] += counts[author]
    if any(not authors for authors in bins):
        raise ValueError("Not enough authors for inner cross-fitting")
    for authors in bins:
        mask = table.iloc[fit].author.isin(authors).to_numpy()
        yield np.flatnonzero(~mask), np.flatnonzero(mask)


def checked_dense(bundle):
    dense, metadata = load_frozen_embeddings(bundle["history"], Path(bundle["dense_path"]).parent)
    if metadata != bundle["identity"]["encoder"]:
        raise ValueError("Frozen encoder artifact changed after prepare")
    return dense


def make_base(bundle, output, fold):
    fit, val = indices(bundle, fold)
    table, history = bundle["table"], bundle["history"]
    dense = checked_dense(bundle)
    context = FeatureContext(table, history, fit, bundle["config"], dense)
    cols = list(dict.fromkeys(c for values in bundle["groups"].values() for c in values))
    frame = table[KEYS + ["content_type"] + cols].copy()
    # Large float32 LU in randomized SVD crashed in multithreaded OpenBLAS on
    # the deployment CPUs. Keep the same SVD recipe with a serial BLAS kernel.
    with threadpool_limits(limits=1, user_api="blas"):
        svd = context.vector_values("svd")[context.history_positions]
    e5 = np.asarray(dense[context.history_positions])
    value = {"run_id": bundle["run_id"], "frame": frame, "columns": cols,
             "fit": fit, "val": val, "svd": svd, "e5": e5,
             "svd_transformer": context.svd}
    atomic_dump(value, output / feature_path(fold, "base"))
    print(f"Base/SVD/E5 ready fold={fold}", flush=True)


def make_extra(bundle, output, fold, block, device="cpu"):
    fit, val = indices(bundle, fold)
    table, history = bundle["table"], bundle["history"]
    needed = np.concatenate([fit, val])
    if block == "org":
        from kairos.experiment_org import organization_features
        values = organization_features(history, table.iloc[needed],
                                       set(table.iloc[fit].author))
    elif block in ("knn_all", "knn180"):
        from kairos.experiment_knn import knn_features
        vectors = checked_dense(bundle)
        kwargs = {"k": 20, "window_days": 180 if block == "knn180" else None,
                  "device": device}
        frames = []
        inner_folds = 3 if fold == "final" else 2
        for a, b in stable_inner_folds(table, fit, inner_folds):
            bank_idx, query_idx = fit[a], fit[b]
            assert set(table.iloc[bank_idx].author).isdisjoint(table.iloc[query_idx].author)
            inner = knn_features(history, vectors, table.iloc[query_idx], table.iloc[bank_idx], **kwargs)
            inner.index = query_idx
            frames.append(inner)
        heldout = knn_features(history, vectors, table.iloc[val], table.iloc[fit], **kwargs)
        heldout.index = val
        frames.append(heldout)
        values = pd.concat(frames).loc[needed]
    else:
        raise ValueError(block)
    if not values[KEYS].reset_index(drop=True).equals(table.iloc[needed][KEYS].reset_index(drop=True)):
        raise ValueError("Feature key alignment changed")
    cols = [c for c in values if c not in KEYS]
    frame = pd.DataFrame(np.nan, index=table.index, columns=cols)
    frame.loc[needed, cols] = values[cols].to_numpy()
    atomic_dump({"run_id": bundle["run_id"], "frame": frame, "columns": cols,
                 "fit_author_sha256": digest(sorted(set(table.iloc[fit].author))),
                 "query_key_sha256": digest(table.iloc[needed][KEYS].values.tolist()),
                 "inner_folds": (3 if fold == "final" else 2) if block.startswith("knn") else 0},
                output / feature_path(fold, block))
    print(f"Feature ready fold={fold} block={block}", flush=True)


def model_input(bundle, output, fold, family, block):
    base = load_artifact(output, feature_path(fold, "base"), bundle)
    frame, cols = base["frame"].copy(), base["columns"].copy()
    if family != "A":
        values = base["e5" if family == "B" else "svd"]
        names = [f"vector_{j}" for j in range(values.shape[1])]
        frame = pd.concat([frame, pd.DataFrame(values, columns=names, index=frame.index)], axis=1)
        cols += names
    if block != "none":
        for component in block.split("+"):
            extra = load_artifact(output, feature_path(fold, component), bundle)
            frame = pd.concat([frame, extra["frame"]], axis=1)
            cols += extra["columns"]
    return frame, cols, base


def job_name(family, block, param):
    return f"{family}__{block}__{param}"


def fit_job(output, fold, family, block, param, threads=4):
    bundle = load_bundle(output)
    name = job_name(family, block, param)
    target = output / "jobs" / str(fold) / f"{name}.joblib"
    if target.exists():
        previous = joblib.load(target)
        if previous.get("run_id") == bundle["run_id"]:
            return name
    start = time.monotonic()
    with threadpool_limits(limits=threads):
        frame, cols, base = model_input(bundle, output, str(fold), family, block)
        fit, val = base["fit"], base["val"]
        config = {**bundle["config"], "hgb": parameter_grid()[param]}
        model = numeric_model(cols, config)
        y = bundle["table"].iloc[fit].label.astype(int)
        model.fit(frame.iloc[fit], y)
        probabilities = model.predict_proba(frame.iloc[val])
    value = {"run_id": bundle["run_id"], "name": name, "family": family, "block": block,
             "param": param, "fold": str(fold), "val": val, "probabilities": probabilities,
             "seconds": time.monotonic() - start,
             "iterations": int(model.named_steps["clf"].n_iter_)}
    if str(fold) != "final":
        value["metrics"] = metrics(bundle["table"].iloc[val].label.astype(int), probabilities.argmax(1))
    else:
        atomic_dump({"model": model, "svd": base["svd_transformer"], "columns": cols},
                    output / "final" / f"{name}.joblib")
    atomic_dump(value, target)
    print(f"FIT {fold} {name} {value.get('metrics',{}).get('macro_f1')} {value['seconds']:.1f}s", flush=True)
    return name


def fit_phase(output, fold, phase, jobs=4, threads=4):
    bundle = load_bundle(output)
    tasks = []
    if phase in ("screen", "controls"):
        for family in "ABC":
            tasks += [(family, "none", p) for p in ("base", "leaves15", "min50", "l2_1")]
            if phase == "screen":
                tasks += [(family, b, "base") for b in bundle["config"].get("feature_blocks", [])]
    elif phase == "combo":
        chosen = load_decision(output, "feature_selection.json", bundle)
        tasks = [(f, b, "base") for f, b in chosen["combo_candidates"].items() if b != "none"]
    elif phase == "tune":
        chosen = load_decision(output, "feature_selection.json", bundle)
        tasks = [(f, chosen["blocks"][f], p) for f in "ABC" for p in parameter_grid()]
    else:
        raise ValueError(phase)
    Parallel(n_jobs=jobs)(delayed(fit_job)(output, fold, f, b, p, threads) for f, b, p in tasks)


def collect(output, bundle, family, block, param):
    table = bundle["table"]
    train_idx = np.flatnonzero(table.split.eq("train"))
    reverse = np.full(len(table), -1, dtype=int)
    reverse[train_idx] = np.arange(len(train_idx))
    probs = np.full((len(train_idx), 3), np.nan)
    fold_ids = np.empty(len(train_idx), dtype=int)
    for fold in range(3):
        path = Path("jobs") / str(fold) / f"{job_name(family,block,param)}.joblib"
        if not (output / path).exists():
            return None
        value = load_artifact(output, path, bundle)
        expected_val = bundle["folds"][fold][1]
        if not np.array_equal(value["val"], expected_val):
            raise ValueError("OOF keys/order do not match the declared fold")
        rel = reverse[value["val"]]
        if (rel < 0).any():
            raise ValueError("Test rows in validation")
        probs[rel] = value["probabilities"]
        fold_ids[rel] = fold
    if not np.isfinite(probs).all() or not np.allclose(probs.sum(1), 1):
        raise ValueError("Incomplete OOF predictions")
    return probs, fold_ids


def select_features(output, finish=False):
    bundle = load_bundle(output)
    train = bundle["table"].query("split == 'train'")
    y = train.label.astype(int).to_numpy()
    blocks, checks, combos = {}, {}, {}
    for family in "ABC":
        reference, fold_ids = collect(output, bundle, family, "none", "base")
        choices = [("none", metrics(y, reference.argmax(1))["macro_f1"])]
        accepted = []
        for block in bundle["config"].get("feature_blocks", []):
            result = collect(output, bundle, family, block, "base")
            if result is None:
                checks[f"{family}/{block}"] = {"accepted": False, "reason": "incomplete"}
                continue
            p, _ = result
            gate = passes_gate(y, p, reference, fold_ids)
            checks[f"{family}/{block}"] = gate
            if gate["accepted"]:
                score = metrics(y, p.argmax(1))["macro_f1"]
                choices.append((block, score)); accepted.append((block, score))
        knns = sorted([(b,s) for b,s in accepted if b.startswith("knn")], key=lambda x:-x[1])
        combo = "org+"+knns[0][0] if knns and any(b == "org" for b,s in accepted) else "none"
        combos[family] = combo
        if finish and combo != "none":
            result = collect(output, bundle, family, combo, "base")
            if result is not None:
                p, _ = result
                best_block = max(choices, key=lambda x:x[1])[0]
                best_p, _ = collect(output, bundle, family, best_block, "base")
                gate = passes_gate(y,p,best_p,fold_ids)
                checks[f"{family}/{combo}"] = gate
                if gate["accepted"]:choices.append((combo,metrics(y,p.argmax(1))["macro_f1"]))
        blocks[family] = max(choices, key=lambda x:x[1])[0]
    report = {"run_id":bundle["run_id"],"blocks":blocks,"checks":checks,"combo_candidates":combos,
              "test_labels_used":False,"feature_selection_complete":finish}
    save_json(output / "feature_selection.json", report)
    print(json.dumps(report), flush=True)
    return report


def select_model(output):
    bundle = load_bundle(output)
    table = bundle["table"]; train = table[table.split.eq("train")]
    y = train.label.astype(int).to_numpy()
    chosen = load_decision(output, "feature_selection.json", bundle)
    families, candidates, incomplete = {}, [], []
    for family in "ABC":
        for param in parameter_grid():
            result = collect(output, bundle, family, chosen["blocks"][family], param)
            if result is None:
                incomplete.append(job_name(family,chosen["blocks"][family],param))
                continue
            p, fold_ids = result
            score = metrics(y,p.argmax(1))
            candidate = {"family":family,"block":chosen["blocks"][family],"param":param,**score}
            candidates.append(candidate)
            if family not in families or score["macro_f1"]>families[family]["macro_f1"]:
                families[family]={**candidate,"probabilities":p}
    if set(families)!=set("ABC"):raise ValueError("Missing complete model family")
    blends=[]
    for weights in simplex_weights():
        p=sum(weights[f]*families[f]["probabilities"] for f in "ABC")
        blends.append({"weights":weights,**metrics(y,p.argmax(1))})
    best=max(blends,key=lambda x:x["macro_f1"])
    best_p=sum(best["weights"][f]*families[f]["probabilities"] for f in "ABC")
    a,_=collect(output,bundle,"A","none","base")
    b,_=collect(output,bundle,"B","none","base")
    reference=.25*a+.75*b
    gate=passes_gate(y,best_p,reference,fold_ids)
    if not gate["accepted"]:
        selected={"A":{"family":"A","block":"none","param":"base","weight":.25},
                  "B":{"family":"B","block":"none","param":"base","weight":.75}}
        best_p=reference
    else:
        selected={f:{"family":f,"block":families[f]["block"],"param":families[f]["param"],"weight":w}
                  for f,w in best["weights"].items() if w>0}
    uncertainty=paired_author_bootstrap(y,best_p.argmax(1),reference.argmax(1),train.author.to_numpy(),
                                       n_resamples=1000,seed=42)
    report={"run_id":bundle["run_id"],"selected":selected,"candidate_gate":gate,
            "metrics":metrics(y,best_p.argmax(1)),"reference_metrics":metrics(y,reference.argmax(1)),
            "best_candidate":best,"candidates":candidates,"blends":blends,
            "incomplete_candidates":incomplete,
            "fold_metrics":{str(f):metrics(y[fold_ids==f],best_p[fold_ids==f].argmax(1)) for f in range(3)},
            "paired_author_bootstrap":uncertainty,"test_labels_used":False,
            "interpretation":"Selection score on reused author folds; bootstrap excludes search uncertainty"}
    save_json(output/"selection.json",report)
    pd.DataFrame(candidates).to_csv(output/"tuning.csv",index=False)
    np.savez_compressed(output/"oof_probabilities.npz",selected=best_p,reference=reference,
                        y=y,fold=fold_ids,authors=train.author.to_numpy(dtype=str))
    frozen={**bundle["config"],"mode":"fixed","selected":selected,"output":"results/v2"}
    save_json(output/"solution_v2.json",frozen)
    print(f"Selected {selected}; macro-F1 {report['metrics']['macro_f1']:.6f}",flush=True)
    return report


def final_fit(output, selected=None):
    bundle=load_bundle(output)
    if selected is None:selected=load_decision(output,"selection.json",bundle)["selected"]
    fit,val=indices(bundle,"final")
    if bundle["table"].iloc[val].label.notna().any():raise ValueError("Test labels exposed")
    probabilities=np.zeros((len(val),3),dtype=float)
    for recipe in selected.values():
        name=fit_job(output,"final",recipe["family"],recipe["block"],recipe["param"],bundle["config"].get("threads",4))
        value=load_artifact(output,Path("jobs/final")/f"{name}.joblib",bundle)
        probabilities+=recipe["weight"]*value["probabilities"]
    target=output/"final";target.mkdir(exist_ok=True,parents=True)
    predictions=bundle["table"].iloc[val][KEYS].assign(label=probabilities.argmax(1))
    predictions.to_csv(target/"predictions.csv",index=False)
    np.save(target/"probabilities.npy",probabilities,allow_pickle=False)
    save_json(target/"manifest.json",{"run_id":bundle["run_id"],"selected":selected,
              "train_rows":len(fit),"test_rows":len(val),"test_labels_used":False,
              "predictions_sha256":hashlib.sha256((target/"predictions.csv").read_bytes()).hexdigest()})
    print(f"FINAL {len(val)} predictions saved",flush=True)


def reproduce(data_dir, output, config):
    if config.get("mode")!="fixed" or not config.get("selected"):
        raise ValueError("Use worker search commands for tuning; public pipeline requires frozen selected recipes")
    with threadpool_limits(limits=config.get("threads",4)):
        bundle=prepare(Path(data_dir),Path(output),config)
        make_base(bundle,output,"final")
        blocks={b for r in config["selected"].values() for b in r["block"].split("+") if b!="none"}
        for block in sorted(blocks):make_extra(bundle,output,"final",block,config.get("knn_device","cpu"))
        final_fit(output,config["selected"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action",required=True,choices=["prepare","base","extra","fit","select-features","select-combos","select","final"])
    parser.add_argument("--config",type=Path,default=ROOT/"configs/stage2_search.json")
    parser.add_argument("--data",type=Path,default=ROOT/"data")
    parser.add_argument("--output",type=Path,default=ROOT/"results/v2")
    parser.add_argument("--fold",default="0")
    parser.add_argument("--block",default="org")
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--phase",default="screen")
    parser.add_argument("--jobs",type=int,default=4)
    parser.add_argument("--threads",type=int,default=4)
    args=parser.parse_args()
    with threadpool_limits(limits=args.threads):
        if args.action=="prepare":prepare(args.data,args.output,json.loads(args.config.read_text()))
        elif args.action=="base":make_base(load_bundle(args.output),args.output,args.fold)
        elif args.action=="extra":make_extra(load_bundle(args.output),args.output,args.fold,args.block,args.device)
        elif args.action=="fit":fit_phase(args.output,args.fold,args.phase,args.jobs,args.threads)
        elif args.action=="select-features":select_features(args.output)
        elif args.action=="select-combos":select_features(args.output,True)
        elif args.action=="select":select_model(args.output)
        elif args.action=="final":final_fit(args.output)


if __name__=="__main__":main()
