"""Lab 1 solution: baseline features + richer author history + text (TF-IDF).

    python solution.py                 # data in data/
    python solution.py --data D:/kairos
    python solution.py --no-text       # skip the text model (faster)

Works with a partial release: needs only posts/ and split.json. If labels.csv
is missing, labels are recomputed from the posts (kairos.label.relabel); if
authors/ is missing, follower counts are set to 0.

Writes results/solution.json and results/predictions/solution.csv
(author,post,label) in the format evaluate.py reads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) // 2)))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.sparse import hstack

import kairos.data as kd
from kairos import features
from kairos.label import TYPICAL, author_level, relabel
from kairos.metrics import score

SEED = 42
RESULTS = Path(__file__).resolve().parent / "results"

# Allow a partial release (no labels.csv / authors/).
kd._release = lambda d: Path(d)

EXTRA_NUM = ["last1_rel", "last3_rel", "last5_rel", "hist_std", "hist_max_rel",
             "hist_min_rel", "trend5", "posts_30d", "text_len_vs_author"]
TEXT_NUM = ["p_below", "p_typical", "p_above"]


def load_table(data: Path) -> pd.DataFrame:
    posts = kd.load_posts(data)
    split = json.loads((data / "split.json").read_text(encoding="utf-8"))
    if (data / "labels.csv").is_file():
        labels = pd.read_csv(data / "labels.csv")
    else:
        print("labels.csv not found: recomputing labels from posts", flush=True)
        lab = relabel(posts, split["train"])
        train_set = set(split["train"])
        labels = lab[["author", "post", "label"]].copy()
        labels["split"] = np.where(labels["author"].isin(train_set), "train", "test")
    try:
        followers = kd.load_followers(data) if (data / "authors").is_dir() else pd.Series(dtype=float)
    except Exception:
        followers = pd.Series(dtype=float)

    levelled = author_level(posts)
    feats = features.build(levelled, followers)
    feats = add_history(feats)
    return labels.merge(feats, on=["author", "post"], how="left", validate="one_to_one")


def add_history(df: pd.DataFrame) -> pd.DataFrame:
    """Author-history features computed from EARLIER posts only (no leakage)."""
    g = df.groupby("author")["log_reactions"]
    prev = g.shift(1)
    lvl = df["level"]
    df["last1_rel"] = prev - lvl
    df["last3_rel"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean()) - lvl
    df["last5_rel"] = g.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean()) - lvl
    df["hist_std"] = g.transform(lambda s: s.shift(1).expanding(min_periods=2).std())
    df["hist_max_rel"] = g.transform(lambda s: s.shift(1).expanding().max()) - lvl
    df["hist_min_rel"] = g.transform(lambda s: s.shift(1).expanding().min()) - lvl
    df["trend5"] = df["last5_rel"] - g.transform(
        lambda s: s.shift(6).rolling(5, min_periods=1).mean()).sub(lvl)
    # posting frequency: own posts in the previous 30 days
    ts = df["posted_at"].astype("int64") // 10**9
    def _cnt(s):
        v = s.to_numpy()
        return pd.Series(np.searchsorted(v, v, side="left")
                         - np.searchsorted(v, v - 30 * 86400, side="left"), index=s.index)
    df["posts_30d"] = ts.groupby(df["author"]).transform(_cnt)
    mean_len = df.groupby("author")["n_chars"].transform(
        lambda s: s.shift(1).expanding().mean())
    df["text_len_vs_author"] = np.log1p(df["n_chars"]) - np.log1p(mean_len.fillna(0))
    return df


def text_probs(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold TF-IDF + logistic regression class probabilities."""
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000,
                           sublinear_tf=True, lowercase=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                           max_features=200_000, sublinear_tf=True)
    tr_text, te_text = train["text"].fillna(""), test["text"].fillna("")
    Xtr = hstack([word.fit_transform(tr_text), char.fit_transform(tr_text)]).tocsr()
    Xte = hstack([word.transform(te_text), char.transform(te_text)]).tocsr()
    y = train["label"].to_numpy()

    def lr():
        return LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced")

    oof = np.zeros((len(train), 3))
    for k, (a, b) in enumerate(GroupKFold(n_splits=5).split(Xtr, y, train["author"])):
        oof[b] = lr().fit(Xtr[a], y[a]).predict_proba(Xtr[b])
        print(f"  text fold {k + 1}/5 done", flush=True)
    te = lr().fit(Xtr, y).predict_proba(Xte)
    print("  text-only (test) macro-F1: "
          f"{score(test['label'], te.argmax(1))['macro_f1']:.4f}", flush=True)
    return oof, te


def gbm(numeric: list[str]) -> Pipeline:
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20),
             features.CONTENT_CAT),
        ])),
        ("clf", HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05,
                                               max_leaf_nodes=31, l2_regularization=1.0,
                                               class_weight="balanced",
                                               random_state=SEED)),
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=kd.DATA_DIR)
    ap.add_argument("--no-text", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    data = Path(args.data)

    table = load_table(data)
    table = table[table["level"].notna()]
    train = table[table["split"] == "train"].copy()
    test = table[table["split"] == "test"].copy()
    print(f"train {len(train):,} / test {len(test):,}", flush=True)

    base_num = features.CONTENT_NUM + features.HISTORY_NUM + features.ARTEFACT_NUM
    numeric = base_num + EXTRA_NUM
    if not args.no_text:
        print("text model ...", flush=True)
        oof, te = text_probs(train, test)
        train[TEXT_NUM], test[TEXT_NUM] = oof, te
        numeric = numeric + TEXT_NUM

    results = {}
    pred_base = gbm(base_num).fit(train, train["label"]).predict(test)
    results["baseline_features"] = score(test["label"], pred_base)
    model = gbm(numeric).fit(train, train["label"])
    proba = model.predict_proba(test)
    pred = proba.argmax(1)
    results["solution"] = score(test["label"], pred)
    results["always_typical"] = score(test["label"], np.full(len(test), TYPICAL))

    print(f"\n{'':20s} {'macro-F1':>9s} {'QWK':>7s} {'accuracy':>9s}")
    for name, m in results.items():
        print(f"{name:20s} {m['macro_f1']:9.4f} {m['qwk']:7.4f} {m['accuracy']:9.4f}")

    (RESULTS / "predictions").mkdir(parents=True, exist_ok=True)
    test[["author", "post"]].assign(label=pred).to_csv(
        RESULTS / "predictions" / "solution.csv", index=False)
    (RESULTS / "solution.json").write_text(json.dumps(
        {"features": numeric, "n_train": len(train), "n_test": len(test),
         "results": results}, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote results/predictions/solution.csv ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
