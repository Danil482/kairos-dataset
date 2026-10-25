"""Tier 3 features: CLIP ViT-B/32 image embeddings reduced + author-relative.

Only ~16 349 of 89 018 labelled posts (~18%) have a post image. For the rest
we fill with zeros and flip `has_image=0`.

Features:
    has_image            — 0/1 flag
    clip_pca_00..47      — PCA-48 of 512-dim CLIP embeddings (fit on train)
    img_cos_to_author    — cos(current image, mean of author's past images)
    img_l2_to_author     — L2 distance to that mean
    img_author_count     — how many prior post images the author had
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CLIP_PATH = Path("cache/image_embeddings_clip.parquet")
CLIP_DIM = 512
PCA_N = 48

PCA_COLS = [f"clip_pca_{i:02d}" for i in range(PCA_N)]
IMG_SUMMARY = ["has_image", "img_cos_to_author", "img_l2_to_author",
               "img_author_count"]

TIER3_NUMERIC = IMG_SUMMARY + PCA_COLS


def _author_cos_l2(emb_df: pd.DataFrame, order_df: pd.DataFrame) -> pd.DataFrame:
    """For each post in order_df (sorted by author,time), compute cos / L2
    between the post's image embedding and the mean of PRIOR image embeddings
    for that author. Rows without an image get 0.
    """
    cols = [c for c in emb_df.columns if c.startswith("c")]
    E = emb_df[cols].values.astype(np.float32)
    key_to_row = pd.Series(np.arange(len(emb_df)),
                           index=pd.MultiIndex.from_frame(emb_df[["author", "post"]]))
    rows = key_to_row.reindex(
        pd.MultiIndex.from_frame(order_df[["author", "post"]])
    ).values  # -1 if no image

    n = len(order_df)
    cos = np.zeros(n, dtype=np.float32)
    l2 = np.zeros(n, dtype=np.float32)
    count = np.zeros(n, dtype=np.int32)
    has = np.zeros(n, dtype=np.int8)

    prev_author = None
    running_sum = None
    running_n = 0
    for i, (idx_val, auth) in enumerate(zip(rows, order_df["author"].values)):
        if auth != prev_author:
            running_sum = np.zeros(CLIP_DIM, dtype=np.float32)
            running_n = 0
            prev_author = auth
        has_row = not pd.isna(idx_val)
        idx = int(idx_val) if has_row else -1
        has[i] = 1 if has_row else 0
        count[i] = running_n
        if has_row and running_n > 0:
            e = E[idx]
            mean = running_sum / running_n
            m_norm = float(np.linalg.norm(mean))
            e_norm = float(np.linalg.norm(e))
            if m_norm > 0 and e_norm > 0:
                cos[i] = float(np.dot(e, mean) / (m_norm * e_norm))
            l2[i] = float(np.linalg.norm(e - mean))
        if has_row:
            running_sum += E[idx]
            running_n += 1

    return pd.DataFrame({
        "author": order_df["author"].values,
        "post": order_df["post"].values,
        "has_image": has,
        "img_cos_to_author": cos,
        "img_l2_to_author": l2,
        "img_author_count": count,
    })


def build(tier1_df: pd.DataFrame, train_authors: set,
          pca_n: int = PCA_N) -> pd.DataFrame:
    if not CLIP_PATH.exists():
        raise SystemExit(f"missing {CLIP_PATH} — run the Colab notebook first")
    emb_df = pd.read_parquet(CLIP_PATH)
    emb_df = emb_df[emb_df.get("_ok", True)].drop(columns=["_ok"], errors="ignore")

    order_df = (tier1_df.sort_values(["author", "posted_at"], kind="mergesort")
                [["author", "post"]].reset_index(drop=True))
    cos_l2 = _author_cos_l2(emb_df, order_df)

    # PCA on the posts that have images, fit on train authors only
    emb_df_idx = emb_df.set_index(["author", "post"])
    tier1_keys = pd.MultiIndex.from_frame(tier1_df[["author", "post"]])
    E_aligned = emb_df_idx.reindex(tier1_keys)
    cols = [c for c in E_aligned.columns if c.startswith("c")]
    E_mat = E_aligned[cols].values.astype(np.float32)
    train_mask = tier1_df["author"].isin(train_authors).values
    has_img_mask = ~np.isnan(E_mat[:, 0])

    from sklearn.decomposition import PCA
    pca = PCA(n_components=pca_n, random_state=42)
    pca.fit(E_mat[train_mask & has_img_mask])

    pca_mat = np.zeros((len(tier1_df), pca_n), dtype=np.float32)
    if has_img_mask.any():
        pca_mat[has_img_mask] = pca.transform(E_mat[has_img_mask]).astype(np.float32)

    pca_df = pd.DataFrame(pca_mat, columns=PCA_COLS[:pca_n])
    pca_df.insert(0, "post", tier1_df["post"].values)
    pca_df.insert(0, "author", tier1_df["author"].values)

    out = cos_l2.merge(pca_df, on=["author", "post"], how="left")
    return out


def materialise(tier1_cache: str = "cache/tier1_features.parquet",
                split_path: str = "data/split.json",
                out_path: str = "cache/tier3_features.parquet",
                force: bool = False) -> pd.DataFrame:
    out_path = Path(out_path)
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)
    tier1 = pd.read_parquet(tier1_cache)
    split = json.loads(Path(split_path).read_text())
    df = build(tier1, set(split["train"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


if __name__ == "__main__":
    df = materialise(force=True)
    print(f"wrote cache/tier3_features.parquet: {len(df):,} rows, {df.shape[1]} cols")
    print(f"has_image coverage: {df['has_image'].mean()*100:.1f}%")
    print(f"img_author_count max: {df['img_author_count'].max()}, "
          f"median non-zero: {df.loc[df['img_author_count']>0, 'img_author_count'].median()}")
