import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.metrics import f1_score

from kairos import features
from kairos.data import DATA_DIR, load_followers, load_labels, load_posts
from kairos.label import author_level
from kairos.metrics import score

SEED = 42
RESULTS = Path("./results")
EXTRACTED_DIR = Path("./extracted_features")


def predict_with_weights(probs, weights):
    weighted_probs = probs * weights
    return np.argmax(weighted_probs, axis=1)


def loss_func(weights, probs, y_true):
    preds = predict_with_weights(probs, weights)
    return -f1_score(y_true, preds, average="macro")


def main() -> None:
    t0 = time.time()

    labels = load_labels(args.data if 'args' in locals() else DATA_DIR)
    levelled = author_level(load_posts(args.data if 'args' in locals() else DATA_DIR))
    table = labels.merge(features.build(levelled, load_followers(args.data if 'args' in locals() else DATA_DIR)),
                         on=["author", "post"], how="left", validate="one_to_one")

    tfidf_matrix = np.load(EXTRACTED_DIR / "fast_text_embeddings.npy")
    with open(EXTRACTED_DIR / "fast_text_posts.json", "r", encoding="utf-8") as f:
        text_posts = json.load(f)
    tfidf_cols = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
    df_tfidf = pd.DataFrame(tfidf_matrix, columns=tfidf_cols)
    df_tfidf["post"] = text_posts
    table = table.merge(df_tfidf, on="post", how="left")

    with open(EXTRACTED_DIR / "fast_image_features.json", "r", encoding="utf-8") as f:
        image_features_map = json.load(f)
    img_cols = ["img_width", "img_height", "img_aspect_ratio", "img_brightness", "img_contrast"]
    img_rows = [image_features_map.get(post_urn, [0.0, 0.0, 0.0, 0.0, 0.0]) for post_urn in table["post"]]
    df_img = pd.DataFrame(img_rows, columns=img_cols, index=table.index)
    table = pd.concat([table, df_img], axis=1)

    train, test = table[table["split"] == "train"].copy(), table[table["split"] == "test"].copy()

    numeric_features = features.CONTENT_NUM + features.HISTORY_NUM + features.ARTEFACT_NUM + tfidf_cols + img_cols
    categorical_features = features.CONTENT_CAT

    for col in categorical_features:
        train[col] = train[col].astype(str).fillna("missing")
        test[col] = test[col].astype(str).fillna("missing")

    all_features = numeric_features + categorical_features
    X_train, y_train = train[all_features], train["label"].astype(int).values
    X_test, y_test = test[all_features], test["label"].astype(int).values

    classes = np.unique(y_train)
    weights = len(y_train) / (len(classes) * np.bincount(y_train))
    base_class_weights = dict(zip(classes, weights))

    print("Basic CatBoost model...")
    model = CatBoostClassifier(
        iterations=2500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=8,
        loss_function="MultiClass",
        class_weights=base_class_weights,
        cat_features=categorical_features,
        random_seed=SEED,
        verbose=500,
        task_type="CPU"
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=100)

    train_probs = model.predict_proba(X_train)
    test_probs = model.predict_proba(X_test)

    print("Optimising Macro-F1...")
    init_weights = [1.0, 1.0, 1.0]
    bounds = [(0.1, 10.0), (0.1, 10.0), (0.1, 10.0)]

    res = minimize(loss_func, init_weights, args=(train_probs, y_train), method="Nelder-Mead", bounds=bounds)
    best_weights = res.x
    print(f"Best weights: {best_weights}")

    final_test_preds = predict_with_weights(test_probs, best_weights)

    results = {"catboost_with_threshold_tuning": score(y_test, final_test_preds)}

    print(f"\n{'Model':35s} {'macro-F1':>9s} {'QWK':>7s} {'accuracy':>9s}")
    for name, m in results.items():
        print(f"{name:35s} {m['macro_f1']:9.4f} {m['qwk']:7.4f} {m['accuracy']:9.4f}")

    (RESULTS / "predictions").mkdir(parents=True, exist_ok=True)
    test[["author", "post"]].assign(label=final_test_preds).to_csv(
        RESULTS / "predictions" / "catboost_ultimate.csv", index=False)


if __name__ == "__main__":
    main()
