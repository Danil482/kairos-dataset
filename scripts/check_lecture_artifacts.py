"""Reload saved models and reproduce held-out predictions without retraining."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from scripts.lecture_models import OUT, SOURCE, expanded_text, split_by_plan
from scripts.pilot_models import restrict_support


def main():
    table = joblib.load(SOURCE / "train_table.joblib")["table"]
    audited = pd.read_csv(OUT / "decision_audit" / "oof_predictions.csv")
    checks = []
    for number in (1, 2, 3):
        directory = OUT / f"fold_{number}"
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        if not report["complete"]:
            raise ValueError(f"Fold {number} is incomplete")
        schema = json.loads((directory / "feature_schema.json").read_text(encoding="utf-8"))
        _, _, validation = split_by_plan(table, report["author_plan"])
        positions = np.random.default_rng(20261025).choice(len(validation), 64, replace=False)
        sample = validation.iloc[positions].copy()
        sample["expanded_text"] = expanded_text(sample)
        matrix = joblib.load(directory / "tfidf.joblib").transform(sample.expanded_text)
        probabilities = {}
        for name in ("linear_svm", "naive_bayes", "random_forest", "catboost_800"):
            if name == "catboost_800":
                model = CatBoostClassifier().load_model(str(directory / "catboost_800.cbm"))
                raw = model.predict_proba(sample[schema["catboost_columns"]])
            else:
                model = joblib.load(directory / f"{name}.joblib")
                features = sample[schema["numeric"] + schema["categorical"]] if name == "random_forest" else matrix
                raw = model.predict_proba(features)
            actual = restrict_support(raw, sample, schema["cutoffs"][0])
            with np.load(directory / f"{name}_probabilities.npz") as stored:
                expected = stored["validation"][positions]
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
            probabilities[name] = actual
            checks.append(dict(fold=number, model=name, posts=len(sample), probabilities_match=True))
        config = report["models"]["catboost_svm_fusion"]["calibration"]
        weight = config["svm_weight"]
        actual = (1-weight)*probabilities["catboost_800"] + weight*probabilities["linear_svm"]
        with np.load(directory / "catboost_svm_fusion_probabilities.npz") as stored:
            np.testing.assert_allclose(actual, stored["validation"][positions], rtol=1e-6, atol=1e-8)
        checks.append(dict(fold=number, model="catboost_svm_fusion", posts=len(sample), probabilities_match=True))
        audit = json.loads((OUT / "decision_audit" / f"fold_{number}.json").read_text(encoding="utf-8"))
        expected = sample[["author", "post"]].merge(audited, on=["author", "post"], how="left", validate="one_to_one")
        for name in ("linear_svm", "naive_bayes", "random_forest", "catboost_800", "catboost_svm_fusion"):
            decision = audit["models"][name]["calibration"]
            if name == "catboost_svm_fusion":
                weight = decision["svm_weight"]
                values = (1-weight)*probabilities["catboost_800"] + weight*probabilities["linear_svm"]
            else:
                values = probabilities[name]
            predicted = (values * decision["multipliers"]).argmax(axis=1)
            np.testing.assert_array_equal(predicted, expected[name].to_numpy())
        print(f"Fold {number}: saved models reproduce 64 held-out probabilities and audited decisions", flush=True)
    (OUT / "artifact_checks.json").write_text(json.dumps(dict(checks=checks, audited_decisions_match=True, passed=True), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
