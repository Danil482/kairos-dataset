"""Post-fit aggregate diagnostics, without exposing private content."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scripts.priority_features import OUT, POLICY
from scripts.semantic_embeddings import SOURCE, save_json
from scripts.priority_models import gate_objective, split_gate_authors
from scripts.calibrate_lecture_decisions import select_decision
from train_maria import metrics, bootstrap_authors


def blending_controls():
    parts, details = [], {}
    for number in (1,2,3):
        directory = OUT / f"fold_{number}"
        calibration = pd.read_csv(directory / "calibration_keys.csv")
        validation = pd.read_csv(directory / "predictions.csv")[["author","post","label"]]
        first = np.load(directory / "blend_text_probabilities.npz")
        second = np.load(directory / "blend_visual_probabilities.npz")
        fit = split_gate_authors(calibration)
        x = np.ones((len(calibration),1))
        solution = minimize(gate_objective, np.zeros(1), jac=True, method="L-BFGS-B",
            args=(x[fit], first["calibration"][fit], second["calibration"][fit], calibration.label.to_numpy()[fit], 0.),
            options=dict(maxiter=300, ftol=1e-10))
        if not solution.success:
            raise ValueError("Constant gate optimizer failed")
        weight = float(expit(solution.x[0]))
        cal = (1-weight)*first["calibration"]+weight*second["calibration"]
        val = (1-weight)*first["validation"]+weight*second["validation"]
        decision = select_decision(calibration.label.to_numpy()[~fit], cal[~fit])
        validation["learned_constant"] = (val*decision["multipliers"]).argmax(axis=1)
        old = json.loads((ROOT / "results" / "semantic_history" / f"fold_{number}" / "report.json").read_text())
        rule = old["models"]["visual_neighbors"]["calibration"]["multipliers"]
        validation["fixed_trees_old_decision"] = (second["validation"]*rule).argmax(axis=1)
        prior = np.load(ROOT / "results" / "semantic_history" / f"fold_{number}" / "visual_neighbors_probabilities.npz")
        detail = dict(visual_weight=weight, intercept=float(solution.x[0]), decision=decision,
            old_decision_multipliers=rule, raw_probability_mean_absolute_change=float(np.abs(second["validation"]-prior["validation"]).mean()),
            protocol="Post-planned control: intercept-only gate fitted on the same first half of calibration authors; decision fitted on the second half. No outer-label parameter selection.")
        save_json(directory / "blending_controls.json", detail)
        details[str(number)] = detail
        parts.append(validation)
    return pd.concat(parts, ignore_index=True), details


def main():
    report = json.loads((OUT / "comparison.json").read_text())
    table = joblib.load(SOURCE)["table"]
    features = joblib.load(OUT / "features.joblib")
    if not table[["author", "post"]].reset_index(drop=True).equals(features[["author", "post"]]):
        raise ValueError("Feature alignment differs")
    data = pd.read_csv(OUT / "oof_predictions.csv").merge(features, on=["author", "post"], validate="one_to_one")
    controls, control_details = blending_controls()
    control_data = data[["author","post","label","v2","adaptive_blend","constant_blend","blend_visual"]].merge(controls,
        on=["author","post","label"],validate="one_to_one")
    control_data.to_csv(OUT / "blending_controls_oof.csv", index=False)
    names = ["v2"]+report["models_completed"]
    groups = dict(no_same_type=data.history_type_n.eq(0), sparse_same_type=data.history_type_n.lt(5),
        sparse_same_quote_mode=data.history_quote_n.lt(5), own_without_quote=data.history_quote_mode.eq(0),
        quote_with_comment=data.history_quote_mode.eq(1), quote_without_comment=data.history_quote_mode.eq(2),
        document_text=data.document_text_available.gt(0), past_images=data.nov_image_past_available.gt(0),
        comparable_image_history=data.nov_image_past_available.gt(0)&data.nov_image_current_available.gt(0))
    by_group = {name:dict(n=int(mask.sum()), authors=int(data.loc[mask,"author"].nunique()),
        macro_f1={model:metrics(data.loc[mask,"label"], data.loc[mask,model])["macro_f1"] for model in names}) for name,mask in groups.items()}
    selected = data[["author", "post", "label", "v2"]].copy()
    selected["inner_selected"] = selected.v2
    selection = {}
    for number in (1,2,3):
        fold = json.loads((OUT / f"fold_{number}" / "report.json").read_text())
        old = json.loads((ROOT / "results" / "semantic_history" / f"fold_{number}" / "report.json").read_text())
        # A diagnostic, post-planned selector, not a claim of an untouched prospective experiment.
        options = {"v2":old["models"]["visual_neighbors"]["calibration"]["calibration_macro_f1"]}
        options.update({name:fold["models"][name]["calibration"]["calibration_macro_f1"]
                        for name in POLICY["feature_experiments"]+["balanced_loss","ordinal"]})
        chosen = max(options, key=options.get)
        mask = data.fold.eq(number)
        selected.loc[mask,"inner_selected"] = data.loc[mask,chosen]
        selection[str(number)] = dict(chosen=chosen, inner_macro_f1=options[chosen])
    result = dict(scope="Descriptive diagnostics on development CV, not model selection on official test",
        ordinal_minus_history_quote_95=bootstrap_authors(data,"ordinal","history_quote"),
        separate_fields_minus_history_quote_95=bootstrap_authors(data,"separate_fields","history_quote"),
        blending_controls=dict(folds=control_details,
            oof={name:metrics(control_data.label,control_data[name]) for name in ["learned_constant","fixed_trees_old_decision"]},
            adaptive_minus_learned_constant_95=bootstrap_authors(control_data,"adaptive_blend","learned_constant"),
            fixed_trees_old_decision_minus_v2_95=bootstrap_authors(control_data,"fixed_trees_old_decision","v2")),
        groups=by_group, inner_selected_diagnostic=dict(folds=selection, metrics=metrics(selected.label, selected.inner_selected),
            caveat="Selector diagnostic added during the research run; includes only models using the full inner calibration set. Not a separately pre-registered experiment."),
        error_changes={name:dict(corrected=int(((data.v2!=data.label)&(data[name]==data.label)).sum()),
                                 introduced=int(((data.v2==data.label)&(data[name]!=data.label)).sum())) for name in names[1:]},
        class_recall={name:{key:report["oof"][name]["by_class"][key]["recall"] for key in ("below","typical","above")} for name in names},
        note="Tree feature importance and subgroup correlations are descriptive, not causal evidence.")
    save_json(OUT / "diagnostics.json", result)
    print(json.dumps(dict(groups={key:value["n"] for key,value in by_group.items()}, inner_selected=result["inner_selected_diagnostic"]), indent=2))


if __name__ == "__main__":
    main()
