"""Verify saved research predictions, nested author sets and source provenance."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.priority_features import OUT, freeze_v2
from scripts.priority_models import NAMES
from scripts.semantic_embeddings import fingerprint, save_json
from train_maria import metrics


def main():
    freeze_v2()
    feature_manifest = json.loads((OUT / "features_manifest.json").read_text())
    for file, key in (("features.joblib", "features_sha256"), ("field_vectors.npz", "field_vectors_sha256"), ("fields.joblib", "fields_sha256")):
        if fingerprint(OUT / file) != feature_manifest[key]:
            raise ValueError("Saved feature hash mismatch")
    parts = []
    for number in (1,2,3):
        directory = OUT / f"fold_{number}"
        report = json.loads((directory / "report.json").read_text())
        if not report["complete"] or set(report["models"]) != set(NAMES):
            raise ValueError("Incomplete training")
        author_sets = [set(report["author_plan"][key]) for key in ("fitting_authors", "calibration_authors", "validation_authors")]
        if any(author_sets[i]&author_sets[j] for i in range(3) for j in range(i)):
            raise ValueError("Author leakage between splits")
        cal = pd.read_csv(directory / "calibration_keys.csv")
        val = pd.read_csv(directory / "predictions.csv")
        if set(cal.author) != author_sets[1] or set(val.author) != author_sets[2]:
            raise ValueError("Prediction author sets differ")
        for name in NAMES:
            stored = np.load(directory / f"{name}_probabilities.npz")
            for key, table in (("calibration", cal), ("validation", val)):
                p = stored[key]
                if p.shape != (len(table), 3) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(axis=1), 1):
                    raise ValueError("Invalid probability cache")
            decision = report["models"][name]["calibration"]["multipliers"]
            np.testing.assert_array_equal((stored["validation"]*decision).argmax(axis=1), val[name])
            actual = metrics(val.label, val[name])
            if actual != report["models"][name]["metrics"]:
                raise ValueError("Report metric does not reproduce")
            details = report["models"][name]["details"]
            if name not in ("constant_blend", "adaptive_blend"):
                if not details["roundtrip_passed"] or details["script_sha256_at_fit"] != fingerprint(ROOT / "scripts" / "priority_models.py"):
                    raise ValueError("Model implementation/provenance differs")
                schema = json.loads((directory / f"{name}_schema.json").read_text())
                if set(schema["columns"]) & {"author","post","label","reactions","resid","split"}:
                    raise ValueError("Outcome or identity in feature schema")
            if name.startswith("blend_") and (not details["fixed_budget_without_calibration_labels"] or details["trees"] != 800):
                raise ValueError("Blending base model saw calibration labels")
            if name in ("constant_blend", "adaptive_blend"):
                fit, select = set(details["gate_fitting_authors"]), set(details["gate_selection_authors"])
                if fit&select or fit|select != author_sets[1] or (fit|select)&author_sets[0]:
                    raise ValueError("Invalid blending author split")
        parts.append(val.assign(fold=number))
    oof = pd.read_csv(OUT / "oof_predictions.csv")
    rebuilt = pd.concat(parts, ignore_index=True)
    pd.testing.assert_frame_equal(oof[["author","post","label","fold"]+NAMES], rebuilt[["author","post","label","fold"]+NAMES])
    save_json(OUT / "artifact_audit.json", dict(complete=True, folds=3, variants=12, rows=len(oof),
        all_saved_predictions_and_metrics_reproduced=True, nested_author_sets_disjoint=True,
        fixed_budget_blending_base_models_verified=True, frozen_v2_verified=True,
        sources={path.name:fingerprint(path) for path in sorted((ROOT / "scripts").glob("*priority*.py"))}))
    print("Artifact audit passed: all 36 fold/variant probability caches reproduce predictions and metrics; author isolation verified")


if __name__ == "__main__":
    main()
