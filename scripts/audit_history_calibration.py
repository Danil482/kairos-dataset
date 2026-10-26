"""Separate the effect of text fusion from the effect of decision calibration.

Reuses the saved outer/inner author splits; never uses official test labels.
Does not change the frozen final model or its predictions.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from train_maria import RESULTS, DATA_DIR, CATEGORY, boost, calibrate, combine, load_table, metrics, bootstrap_authors


def main():
    validation = json.loads((RESULTS / "validation.json").read_text())
    table, numeric = load_table(DATA_DIR)
    table = table.loc[table.split == "train"]
    oof = pd.read_csv(RESULTS / "oof_predictions.csv")
    columns = numeric + CATEGORY
    results, configurations = [], []
    for fold in validation["folds"]:
        fitting = table.loc[table.author.isin(fold["fitting_authors"])]
        calibration = table.loc[table.author.isin(fold["calibration_authors"])]
        heldout = table.loc[table.author.isin(fold["validation_authors"])]
        model = boost(validation["iterations"]).fit(fitting[columns], fitting.label, cat_features=CATEGORY)
        probability = model.predict_proba(calibration[columns])
        config = calibrate(calibration.label, probability, probability)
        probability = model.predict_proba(heldout[columns])
        result = heldout[["author", "post"]].copy()
        result["reproduced_history"] = probability.argmax(axis=1)
        result["history_calibrated"] = combine(probability, probability, config).argmax(axis=1)
        results.append(result)
        configurations.append(dict(fold=fold["fold"], config=config))
        print(f"Calibrated history audit: fold {fold['fold']} complete", flush=True)
    oof = oof.merge(pd.concat(results), on=["author", "post"], validate="one_to_one")
    if not np.array_equal(oof.history.to_numpy(), oof.reproduced_history.to_numpy()):
        raise RuntimeError("Original history models did not reproduce; cannot compare calibration fairly")
    report = dict(
        history_calibrated=metrics(oof.label, oof.history_calibrated),
        fusion=metrics(oof.label, oof.fusion),
        fusion_minus_history_calibrated_95=bootstrap_authors(oof, "fusion", "history_calibrated"),
        fold_configurations=configurations,
    )
    oof.to_csv(RESULTS / "oof_calibration_audit.csv", index=False)
    (RESULTS / "calibration_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "fold_configurations"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
