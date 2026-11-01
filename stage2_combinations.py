"""Test both predeclared ORG/KNN combinations without changing prepared workers.

Run --action fit --fold 0 (then 1 and 2), followed by --action select.
Only released-train author OOF predictions participate in selection.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from kairos.experiment_stage2 import (
    ROOT, collect, fit_job, load_bundle, metrics, passes_gate,
    paired_author_bootstrap, save_json, select_features,
)

COMBINATIONS = ("org+knn_all", "org+knn180")


def provenance():
    path = Path(__file__).resolve()
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def fit_combinations(output, fold, jobs=4, threads=4):
    if str(fold) not in ("0", "1", "2"):
        raise ValueError("Combination fitting requires outer fold 0, 1 or 2")
    if jobs < 1 or threads < 1:
        raise ValueError("jobs and threads must be positive")
    output = Path(output)
    # Every worker checks its prepared run identity and reuses feature artifacts.
    return Parallel(n_jobs=jobs)(delayed(fit_job)(
        output, str(fold), family, block, "base", threads)
        for family in "ABC" for block in COMBINATIONS)


def select_combinations(output):
    output = Path(output)
    bundle = load_bundle(output)
    table = bundle["table"]
    if table.loc[table.split.eq("test"), "label"].notna().any():
        raise ValueError("Test labels exposed to combination selection")
    train = table.loc[table.split.eq("train")]
    y, authors = train.label.astype(int).to_numpy(), train.author.to_numpy()
    # Recover the existing, gated individual choices before testing synergy.
    decision = select_features(output, finish=False)
    script = provenance()
    audit = {"run_id": bundle["run_id"], "script": script, "test_labels_used": False,
             "combinations": list(COMBINATIONS), "families": {},
             "interpretation": "Selection on reused author folds; bootstrap excludes search uncertainty"}
    for family in "ABC":
        baseline_result = collect(output, bundle, family, "none", "base")
        individual = decision["blocks"][family]
        individual_result = collect(output, bundle, family, individual, "base")
        if baseline_result is None or individual_result is None:
            raise ValueError(f"Incomplete reference predictions for {family}")
        baseline, fold_ids = baseline_result
        reference, reference_folds = individual_result
        if not np.array_equal(fold_ids, reference_folds):
            raise ValueError("Reference folds do not align")
        baseline_metrics = metrics(y, baseline.argmax(1))
        individual_metrics = metrics(y, reference.argmax(1))
        family_audit = {"individual_choice": individual,
                        "baseline_metrics": baseline_metrics,
                        "individual_metrics": individual_metrics, "candidates": {}}
        selected, selected_score = individual, individual_metrics["macro_f1"]
        for block in COMBINATIONS:
            result = collect(output, bundle, family, block, "base")
            if result is None:
                entry = {"accepted": False, "reason": "incomplete"}
                family_audit["candidates"][block] = entry
                decision["checks"][f"{family}/{block}"] = entry
                continue
            probabilities, candidate_folds = result
            if not np.array_equal(fold_ids, candidate_folds):
                raise ValueError("Combination folds do not align")
            predicted = probabilities.argmax(1)
            score = metrics(y, predicted)
            gate_individual = passes_gate(y, probabilities, reference, fold_ids)
            gate_baseline = passes_gate(y, probabilities, baseline, fold_ids)
            accepted = gate_individual["accepted"] and gate_baseline["accepted"]
            uncertainty_individual = paired_author_bootstrap(
                y, predicted, reference.argmax(1), authors, n_resamples=1000, seed=42)
            uncertainty_baseline = uncertainty_individual if individual == "none" else paired_author_bootstrap(
                y, predicted, baseline.argmax(1), authors, n_resamples=1000, seed=42)
            entry = {"accepted": accepted, "metrics": score,
                     "gate_vs_individual": gate_individual, "gate_vs_baseline": gate_baseline,
                     "fold_metrics": {str(fold): metrics(y[fold_ids == fold], predicted[fold_ids == fold])
                                      for fold in np.unique(fold_ids)},
                     "bootstrap_vs_individual": uncertainty_individual,
                     "bootstrap_vs_baseline": uncertainty_baseline}
            family_audit["candidates"][block] = entry
            decision["checks"][f"{family}/{block}"] = {
                "accepted": accepted, "gate_vs_individual": gate_individual,
                "gate_vs_baseline": gate_baseline}
            if accepted and score["macro_f1"] > selected_score:
                selected, selected_score = block, score["macro_f1"]
        decision["blocks"][family] = selected
        family_audit["selected"] = selected
        audit["families"][family] = family_audit
    decision.update({"run_id": bundle["run_id"], "feature_selection_complete": True,
                     "test_labels_used": False, "combination_script": script,
                     "all_combination_candidates": list(COMBINATIONS),
                     "combination_policy": "Both gates against the chosen individual block and baseline must pass; incomplete candidates excluded"})
    save_json(output / "combo_audit.json", audit)
    save_json(output / "feature_selection.json", decision)
    print(f"Combination selection: {decision['blocks']}", flush=True)
    return decision, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", required=True, choices=("fit", "select"))
    parser.add_argument("--output", type=Path, default=ROOT / "results/v2")
    parser.add_argument("--fold", choices=("0", "1", "2"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.action == "fit":
        if args.fold is None:
            parser.error("--fold is required for --action fit")
        fit_combinations(args.output, args.fold, args.jobs, args.threads)
    else:
        select_combinations(args.output)


if __name__ == "__main__":
    main()
