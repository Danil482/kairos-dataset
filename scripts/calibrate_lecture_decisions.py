"""Select class decision multipliers using inner authors, with a shared broad grid.

The narrow pilot grid hit its upper bound on INNER calibration data for both
text models. Audit a predeclared broader grid for every model, including the
previous 400-tree CatBoost control. Reuse fitted models and probabilities;
preserve the original reports. Outer labels never select multipliers or weights.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.lecture_models import OUT, SOURCE, save_json
from train_maria import bootstrap_authors, metrics

GRID = (.25, .5, .75, 1., 1.25, 1.5, 2., 2.5, 3., 4., 6., 8.)
MODELS = ["linear_svm", "naive_bayes", "random_forest", "catboost_800", "catboost_400"]
DESTINATION = OUT / "decision_audit"


def select_decision(labels, probabilities):
    labels = np.asarray(labels, dtype=int)
    best = None
    for below in GRID:
        for above in GRID:
            multiplier = [below, 1., above]
            prediction = (probabilities * multiplier).argmax(axis=1)
            matrix = np.bincount(3*labels+prediction, minlength=9).reshape(3, 3)
            denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
            score = float(np.divide(2*matrix.diagonal(), denominator,
                          out=np.zeros(3, dtype=float), where=denominator != 0).mean())
            if best is None or score > best["calibration_macro_f1"]:
                best = dict(calibration_macro_f1=score, multipliers=multiplier)
    return best


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    parts, folds = [], []
    for number in (1, 2, 3):
        folder = OUT / f"fold_{number}"
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        if not report["complete"]:
            raise ValueError(f"Fold {number} is incomplete")
        cal_keys = pd.read_csv(folder / "calibration_keys.csv")
        validation = pd.read_csv(folder / "predictions.csv")[["author", "post", "label"]]
        previous = pd.read_csv(SOURCE / f"pilot_fold{number}_predictions.csv")
        if not validation.equals(previous[["author", "post", "label"]]):
            raise ValueError("Previous CatBoost validation order differs")
        if set(cal_keys.author) != set(report["author_plan"]["calibration_authors"]):
            raise ValueError("Calibration population differs")
        models, probabilities = {}, {}
        for name in MODELS:
            file = (SOURCE / f"pilot_fold{number}_native_text_unweighted_probabilities.npz"
                    if name == "catboost_400" else folder / f"{name}_probabilities.npz")
            with np.load(file) as stored:
                cal, val = stored["calibration"], stored["validation"]
            if cal.shape != (len(cal_keys), 3) or val.shape != (len(validation), 3):
                raise ValueError("Probability populations differ")
            config = select_decision(cal_keys.label, cal)
            validation[name] = (val * config["multipliers"]).argmax(axis=1)
            models[name] = dict(calibration=config, metrics=metrics(validation.label, validation[name]))
            probabilities[name] = (cal, val)
        best = None
        for weight in (0., .1, .25, .5):
            cal = (1-weight)*probabilities["catboost_800"][0] + weight*probabilities["linear_svm"][0]
            config = select_decision(cal_keys.label, cal) | {"svm_weight": weight}
            if best is None or config["calibration_macro_f1"] > best["calibration_macro_f1"]:
                best = config
        weight = best["svm_weight"]
        val = (1-weight)*probabilities["catboost_800"][1] + weight*probabilities["linear_svm"][1]
        validation["catboost_svm_fusion"] = (val * best["multipliers"]).argmax(axis=1)
        models["catboost_svm_fusion"] = dict(calibration=best, metrics=metrics(validation.label, validation.catboost_svm_fusion))
        fold = dict(fold=number, models=models, multiplier_grid=GRID)
        folds.append(fold)
        save_json(DESTINATION / f"fold_{number}.json", fold)
        parts.append(validation.assign(fold=number))
        print(f"Fold {number} shared decision grid: " + str({k:v['metrics']['macro_f1'] for k,v in models.items()}), flush=True)
    combined = pd.concat(parts, ignore_index=True)
    reference = pd.read_csv(SOURCE / "all_oof_predictions.csv")
    joined = combined.merge(reference[["author", "post", "label", "v1_fusion", "native_text_unweighted"]],
                            on=["author", "post", "label"], how="outer", validate="one_to_one", indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("Coverage differs from the 71395-post development population")
    combined = joined.drop(columns="_merge")
    names = MODELS + ["catboost_svm_fusion", "v1_fusion", "native_text_unweighted"]
    summary = dict(n=len(combined), authors=int(combined.author.nunique()),
        scope="Three author-held-out folds inside official train; exploratory model selection, not an official test score",
        decision_selection="Shared broad grid on inner calibration authors only; no outer tuning. SVM sigmoid also uses inner authors.",
        multiplier_grid=GRID, svm_weight_grid=[0., .1, .25, .5],
        oof={name:metrics(combined.label, combined[name]) for name in names},
        folds={str(f["fold"]):{k:v["metrics"]["macro_f1"] for k,v in f["models"].items()} for f in folds},
        differences_vs_catboost_400={name:bootstrap_authors(combined, name, "catboost_400") for name in ("catboost_800", "catboost_svm_fusion", "random_forest")},
        differences_vs_original_v2={name:bootstrap_authors(combined, name, "native_text_unweighted") for name in ("catboost_800", "catboost_svm_fusion")},
        interpretation="Bootstrap resamples entire authors and conditions on predictions; it does not account for model-selection uncertainty.")
    summary["best_development_model"] = max(names, key=lambda name:summary["oof"][name]["by_class"]["macro avg"]["f1-score"])
    combined.to_csv(DESTINATION / "oof_predictions.csv", index=False)
    save_json(DESTINATION / "comparison.json", summary)
    print(json.dumps({"oof_macro_f1":{name:summary["oof"][name]["macro_f1"] for name in names},
                      "differences_vs_original_v2":summary["differences_vs_original_v2"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
