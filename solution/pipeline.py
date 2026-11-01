import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from kairos.data import load_posts, load_labels, load_followers
from kairos.label import author_level
from kairos.metrics import score
from .features import (
    build_dense_features, build_text_matrix, load_media_counts,
    load_image_features, build_bm25_features,
)

from .config import (
    SEED, SVD_DIMS, WORD_TFIDF, CHAR_TFIDF,
)
from .model import fit_dense, fit_text, fit_catboost, blend, calibrate_weights

RESULTS = Path("results")
PRED_DIR = RESULTS / "predictions"
CALIB_PATH = RESULTS / "lab1_calibration.json"


def load_data(data_dir):
    print("loading data ...")
    labels = load_labels(data_dir)
    posts = load_posts(data_dir)
    followers = load_followers(data_dir)
    media = load_media_counts(data_dir)
    img = load_image_features(data_dir)
    levelled = author_level(posts)
    print(f"posts {len(posts)}, labels {len(labels)}, media rows {len(media)}, "
          f"image posts {len(img)}")
    return labels, levelled, followers, media, img


def prepare_data(labels, levelled, followers, media, img):
    df, dense_cols = build_dense_features(levelled, followers, media)
    if len(img) > 0:
        df = df.merge(img, on=["author", "post"], how="left")
        img_cols = [c for c in img.columns if c not in ("author", "post")]
        for c in img_cols:
            df[c] = df[c].fillna(0)
        dense_cols = dense_cols + img_cols

    print("building BM25 ...")
    df = build_bm25_features(df)
    dense_cols = dense_cols + ["bm25_max", "bm25_mean", "bm25_resid", "bm25_logreact"]

    df = df.merge(labels[["author", "post", "split", "label"]],
                  on=["author", "post"], how="left")
    df = df[df["label"].notna()].reset_index(drop=True)
    X_dense = df[dense_cols].fillna(0).values
    X_dense = np.nan_to_num(X_dense, nan=0.0, posinf=0.0, neginf=0.0)
    return df, X_dense


def split_authors(df, frac=0.2):
    rng = np.random.default_rng(SEED)
    train_authors = df.loc[df["split"] == "train", "author"].unique()
    rng.shuffle(train_authors)
    n_val = int(frac * len(train_authors))
    val_authors = set(train_authors[:n_val].tolist())

    is_train = (df["split"] == "train").values
    in_val = is_train & df["author"].isin(val_authors).values
    in_tr = is_train & ~df["author"].isin(val_authors).values
    return in_tr, in_val


def three_way_blend(p_dense, p_text, p_cat):
    return 0.40 * p_dense + 0.30 * p_text + 0.30 * p_cat


def validate(df, X_dense):
    in_tr, in_val = split_authors(df)
    print(f"validation: train {in_tr.sum()} / val {in_val.sum()}")

    Xtf_tr, Xtf_val = build_text_matrix(
        df.loc[in_tr, "text"].fillna("").values,
        df.loc[in_val, "text"].fillna("").values,
        WORD_TFIDF, CHAR_TFIDF, SVD_DIMS,
    )
    y_tr = df.loc[in_tr, "label"].astype(int).values
    y_val = df.loc[in_val, "label"].astype(int).values

    print("training dense ...")
    p_dense = fit_dense(X_dense[in_tr], y_tr, X_dense[in_val])
    print("training text ...")
    p_text = fit_text(Xtf_tr, y_tr, Xtf_val)
    print("training catboost ...")
    p_cat = fit_catboost(X_dense[in_tr], y_tr, X_dense[in_val])

    p_val = three_way_blend(p_dense, p_text, p_cat)
    raw = score(y_val, p_val.argmax(axis=1))
    print(f"val (argmax):     macro-F1 {raw['macro_f1']:.4f}  QWK {raw['qwk']:.4f}  acc {raw['accuracy']:.4f}")

    print("calibrating ...")
    w, _ = calibrate_weights(p_val, y_val)
    cal = score(y_val, (p_val * w).argmax(axis=1))
    print(f"val (calibrated): macro-F1 {cal['macro_f1']:.4f}  QWK {cal['qwk']:.4f}  acc {cal['accuracy']:.4f}")
    print(f"weights: {w.round(3).tolist()}")

    RESULTS.mkdir(exist_ok=True)
    CALIB_PATH.write_text(json.dumps({"weights": w.tolist()}, indent=2))
    return w


def finalize(df, X_dense, w):
    in_tr = (df["split"] == "train").values
    in_te = (df["split"] == "test").values
    print(f"final: train {in_tr.sum()} / test {in_te.sum()}")

    Xtf_tr, Xtf_te = build_text_matrix(
        df.loc[in_tr, "text"].fillna("").values,
        df.loc[in_te, "text"].fillna("").values,
        WORD_TFIDF, CHAR_TFIDF, SVD_DIMS,
    )
    y_tr = df.loc[in_tr, "label"].astype(int).values
    y_te = df.loc[in_te, "label"].astype(int).values

    print("training dense ...")
    p_dense = fit_dense(X_dense[in_tr], y_tr, X_dense[in_te])
    print("training text ...")
    p_text = fit_text(Xtf_tr, y_tr, Xtf_te)
    print("training catboost ...")
    p_cat = fit_catboost(X_dense[in_tr], y_tr, X_dense[in_te])

    p_te = three_way_blend(p_dense, p_text, p_cat)
    pred = (p_te * w).argmax(axis=1)

    m = score(y_te, pred)
    print(f"test_calibrated  macro-F1 {m['macro_f1']:.4f}  QWK {m['qwk']:.4f}  acc {m['accuracy']:.4f}")

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = df.loc[in_te, ["author", "post"]].copy()
    out["label"] = pred
    out.to_csv(PRED_DIR / "lab1.csv", index=False)
    print(f"wrote {PRED_DIR / 'lab1.csv'}")

    metrics = {
        "test": m,
        "weights": w.tolist(),
        "n_train": int(in_tr.sum()),
        "n_test": int(in_te.sum()),
    }
    (RESULTS / "lab1_metrics.json").write_text(json.dumps(metrics, indent=2))


def run(data_dir, do_validate=True, do_final=True):
    t0 = time.time()
    labels, levelled, followers, media, img = load_data(data_dir)
    df, X_dense = prepare_data(labels, levelled, followers, media, img)

    w = np.array([1.0, 1.0, 1.0])
    if do_validate:
        w = validate(df, X_dense)
    elif CALIB_PATH.exists():
        w = np.array(json.loads(CALIB_PATH.read_text())["weights"])
        print(f"loaded weights {w.round(3).tolist()}")

    if do_final:
        finalize(df, X_dense, w)

    print(f"done in {time.time() - t0:.0f}s")