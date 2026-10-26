"""Diagnose empty-original errors and separate probability quality from decisions."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from train_maria import metrics,bootstrap_authors
from scripts.semantic_embeddings import SOURCE,save_json
from scripts.lecture_models import split_by_plan

OUT = ROOT / "results" / "empty_text"


def main():
    OUT.mkdir(exist_ok=True)
    table = joblib.load(SOURCE)["table"].reset_index(drop=True)
    visual = joblib.load(ROOT / "results" / "visual" / "metadata.joblib")
    table["has_image"] = visual.visual_available
    refs = [("old_visual","visual","multimodal_catboost"),("current","semantic_history","visual_neighbors")]
    rows,calibration_audit = [],[]
    for number in (1,2,3):
        plan = json.loads((ROOT / "results" / "maria" / f"fold_{number}.json").read_text())
        _,cal,val = split_by_plan(table,plan)
        part = val.copy()
        decision_weights = {}
        inner = dict(fold=number,n_empty=int(cal.original_chars.eq(0).sum()),models={})
        for name,folder,model in refs:
            directory = ROOT / "results" / folder / f"fold_{number}"
            report = json.loads((directory / "report.json").read_text())
            prediction = pd.read_csv(directory / "predictions.csv")
            if not prediction[["author","post","label"]].equals(val[["author","post","label"]].reset_index(drop=True)):
                raise ValueError("Reference prediction order differs")
            with np.load(directory / f"{model}_probabilities.npz") as saved:
                c,v = saved["calibration"],saved["validation"]
            weights = report["models"][model]["calibration"]["multipliers"]
            decision_weights[name] = weights
            part[name] = prediction[model].to_numpy()
            part[name+"_raw"] = v.argmax(1)
            part[name+"_loss"] = -np.log(np.maximum(v[np.arange(len(v)),val.label.to_numpy()],1e-15))
            part[name+"_brier"] = ((v-np.eye(3)[val.label.to_numpy()])**2).sum(axis=1)
            mask = cal.original_chars.eq(0).to_numpy()
            inner["models"][name] = dict(multipliers=weights,raw_macro_f1=metrics(cal.label[mask],c[mask].argmax(1))["macro_f1"],
                adjusted=metrics(cal.label[mask],(c[mask]*weights).argmax(1)),log_loss=float(log_loss(cal.label[mask],c[mask],labels=[0,1,2])))
        part["current_old_decision"] = (v*decision_weights["old_visual"]).argmax(1)
        calibration_audit.append(inner)
        rows.append(part)
    data = pd.concat(rows,ignore_index=True)
    empty = data.original_chars.eq(0)
    masks = dict(empty=empty,has_original=~empty,empty_reshared=empty&data.has_reshared.eq(1),
        empty_not_reshared=empty&data.has_reshared.eq(0),empty_with_images=empty&data.has_image.eq(1),
        empty_without_images=empty&data.has_image.eq(0),empty_document=empty&data.has_document.eq(1),
        empty_no_expanded=empty&data.expanded_text.str.strip().eq(""),empty_short_history=empty&data.hist_n.lt(10))
    slices = {}
    for name,mask in masks.items():
        sub = data[mask]
        slices[name] = dict(n=len(sub),authors=int(sub.author.nunique()),
            true_counts=np.bincount(sub.label,minlength=3).tolist(),
            models={model:dict(adjusted=metrics(sub.label,sub[model]),raw=metrics(sub.label,sub[model+"_raw"]),
                log_loss=float(sub[model+"_loss"].mean()),brier=float(sub[model+"_brier"].mean()),
                predicted_counts=np.bincount(sub[model],minlength=3).tolist()) for model,_,_ in refs})
    sub = data[empty]
    report = dict(scope="Descriptive diagnostics of existing development OOF predictions; not causal evidence",slices=slices,
        empty_current_with_old_decision=metrics(sub.label,sub.current_old_decision),
        empty_current_minus_old_95=bootstrap_authors(sub,"current","old_visual"),calibration=calibration_audit,
        transitions=dict(old_correct_new_wrong=int((sub.old_visual.eq(sub.label)&sub.current.ne(sub.label)).sum()),
                         old_wrong_new_correct=int((sub.old_visual.ne(sub.label)&sub.current.eq(sub.label)).sum())))
    save_json(OUT / "diagnosis.json",report)
    print(json.dumps({"slices":{name:dict(n=v["n"],classes=v["true_counts"],models={m:dict(f1=x["adjusted"]["macro_f1"],raw_f1=x["raw"]["macro_f1"],log_loss=x["log_loss"],predicted=x["predicted_counts"]) for m,x in v["models"].items()}) for name,v in slices.items()},
        "empty_current_minus_old_95":report["empty_current_minus_old_95"],"transitions":report["transitions"]},indent=2))


if __name__ == "__main__":
    main()
