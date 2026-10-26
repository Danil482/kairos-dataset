"""Descriptive train-only novelty slices; never used to select model settings."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import joblib
import numpy as np
import pandas as pd
from train_maria import metrics
from scripts.semantic_embeddings import SOURCE,save_json
from scripts.semantic_history import OUT


def main():
    table = joblib.load(SOURCE)["table"].reset_index(drop=True)
    features = joblib.load(OUT / "features.joblib")
    if not table[["author","post"]].equals(features[["author","post"]]):
        raise ValueError("Feature order changed")
    for prefix in ("own","expanded"):
        if not np.array_equal(features[f"nov_{prefix}_past_available"]+features[f"nov_{prefix}_past_missing"],table.hist_n):
            raise ValueError("Semantic history does not match the strict-past benchmark history")
    data = table[["author","post","label","hist_n","original_chars"]].merge(features,on=["author","post"],validate="one_to_one")
    predictions = pd.read_csv(OUT / "oof_predictions.csv")
    model_names = [name for name in predictions if name not in ["author","post","label","fold"]]
    data = data.merge(predictions,on=["author","post","label"],validate="one_to_one")
    groups = pd.qcut(data.nov_own_novelty_100,4,labels=["Q1 familiar","Q2","Q3","Q4 novel"])
    descriptions = []
    for name in ["Q1 familiar","Q2","Q3","Q4 novel"]:
        subset = data.loc[groups.eq(name)]
        descriptions.append(dict(group=name,n=len(subset),authors=int(subset.author.nunique()),
            novelty_range=[float(subset.nov_own_novelty_100.min()),float(subset.nov_own_novelty_100.max())],
            class_shares={str(k):float(subset.label.eq(k).mean()) for k in (0,1,2)},
            macro_f1={model:metrics(subset.label,subset[model])["macro_f1"] for model in model_names}))
    # Demeaning by author removes a between-author mean difference, not confounding in general.
    valid = data[data.nov_own_novelty_100.notna()].copy()
    centered_x = valid.nov_own_novelty_100-valid.groupby("author").nov_own_novelty_100.transform("mean")
    centered_y = valid.label-valid.groupby("author").label.transform("mean")
    reports = [json.loads((OUT / f"fold_{n}" / "report.json").read_text()) for n in (1,2,3)]
    importance = {}
    for model in model_names:
        if all(model in report["models"] for report in reports):
            names = reports[0]["models"][model]["details"]["feature_importance_by_group"]
            importance[model] = {group:float(np.mean([report["models"][model]["details"]["feature_importance_by_group"][group] for report in reports])) for group in names}
    result = dict(history_lengths_match_all_queries=True,history_range=[int(table.hist_n.min()),int(table.hist_n.max())],
        own_novelty_quantiles={str(q):float(valid.nov_own_novelty_100.quantile(q)) for q in (0,.25,.5,.75,1)},
        quartiles=descriptions,within_author_novelty_ordinal_label_correlation=float(centered_x.corr(centered_y)),
        mean_feature_importance_by_group=importance,
        interpretation="Exploratory train associations, not causal effects. Quartiles are descriptive, not tuned prediction rules. Tree importances are model-dependent.")
    save_json(OUT / "diagnostics.json",result)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
