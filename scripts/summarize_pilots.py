"""Compare the three development folds with v1 on exactly the same train posts."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

from train_maria import metrics, bootstrap_authors

OUT = ROOT / "results" / "exploration_v2"
MODELS = ["history_balanced", "history_unweighted", "enriched_unweighted", "native_text_unweighted"]


def main():
    parts = []
    folds = []
    for number in [1, 2, 3]:
        fold = json.loads((OUT / f"pilot_fold{number}.json").read_text())
        if len(fold["models"]) != len(MODELS) or "native_minus_enriched_95" not in fold:
            raise RuntimeError(f"Fold {number} is not complete")
        folds.append(fold)
        part = pd.read_csv(OUT / f"pilot_fold{number}_predictions.csv")
        part["fold"] = number
        parts.append(part)
    combined = pd.concat(parts, ignore_index=True)
    old = pd.read_csv(ROOT / "results" / "maria" / "oof_predictions.csv")
    if len(combined) != len(old) or combined.duplicated(["author", "post"]).any():
        raise RuntimeError("Development prediction coverage differs")
    combined = combined.merge(old[["author", "post", "label", "fusion"]], on=["author", "post", "label"], validate="one_to_one")
    if len(combined) != len(old):
        raise RuntimeError("Development labels do not match")
    combined = combined.rename(columns={"fusion": "v1_fusion"})
    columns = MODELS + ["v1_fusion"]
    report = dict(n=len(combined), authors=combined.author.nunique(),
        oof={name:metrics(combined.label, combined[name]) for name in columns},
        folds={str(f["fold"]):{name:f["models"][name]["metrics"]["macro_f1"] for name in MODELS} for f in folds},
        native_minus_v1_95=bootstrap_authors(combined, "native_text_unweighted", "v1_fusion"),
        native_minus_enriched_95=bootstrap_authors(combined, "native_text_unweighted", "enriched_unweighted"),
        enriched_minus_history_95=bootstrap_authors(combined, "enriched_unweighted", "history_unweighted"),
        native_author_bootstrap_95=bootstrap_authors(combined, "native_text_unweighted"),
        interpretation="Development CV on official train authors; not a new official test result. Intervals condition on the fitted predictions.")
    features = joblib.load(OUT / "train_table.joblib")["table"]
    combined = combined.merge(features[["author", "post", "original_chars", "has_reshared"]], on=["author", "post"], validate="one_to_one")
    masks = dict(empty_text=combined.original_chars.eq(0), has_reshared=combined.has_reshared.eq(1),
                 no_reshared=combined.has_reshared.eq(0))
    report["slices"] = {name:dict(n=int(mask.sum()),
        v1=metrics(combined.loc[mask, "label"], combined.loc[mask, "v1_fusion"]),
        native=metrics(combined.loc[mask, "label"], combined.loc[mask, "native_text_unweighted"]))
        for name,mask in masks.items()}
    combined.to_csv(OUT / "all_oof_predictions.csv", index=False)
    (OUT / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"oof":{k:{m:v[m] for m in ["macro_f1", "qwk", "accuracy"]} for k,v in report["oof"].items()},
                      "folds":report["folds"], "native_minus_v1_95":report["native_minus_v1_95"],
                      "slice_macro_f1":{k:{"n":v["n"], "v1":v["v1"]["macro_f1"], "native":v["native"]["macro_f1"]} for k,v in report["slices"].items()}}, indent=2))


if __name__ == "__main__":
    main()
