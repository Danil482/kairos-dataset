"""Tier 2 features: compact text signals derived from multilingual-e5-small.

Given per-post embeddings in cache/text_embeddings_e5.parquet, we build:

  * raw            — the full 384-dim vector (optional; prefix EMB_).
  * cos_to_author  — cosine between the post embedding and the mean of the
                     author's prior post embeddings (expanding, shift(1)).
  * l2_to_author   — same, L2 distance.
  * emb_pca_N      — N principal components of the per-post embedding,
                     fit on train posts only.
  * text_stats     — simple style stats: emoji_count, caps_ratio, exclaim,
                     bullet_count, placeholder_density, avg_word_len.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

EMB_PATH = Path("cache/text_embeddings_e5.parquet")
DIM = 384
PCA_N = 128

TEXT_STATS = ["avg_word_len", "caps_ratio", "exclaim_count", "question_count_stat",
              "emoji_count", "bullet_count", "placeholder_density"]
EMB_SUMMARY = ["cos_to_author", "l2_to_author", "author_posts_so_far"]
PCA_COLS = [f"emb_pca_{i:02d}" for i in range(PCA_N)]

TIER2_NUMERIC = TEXT_STATS + EMB_SUMMARY + PCA_COLS


# ---------- Text-stat regex
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F600-\U0001F64F]"
)
_BULLET_LINE = re.compile(r"(?m)^\s*[-•◦●▪➤→*]\s")
_PLACEHOLDER = re.compile(r"⟨[^⟩]+⟩")


def _text_stats(text: str) -> dict:
    text = text or ""
    words = text.split()
    nw = len(words)
    letters = [c for c in text if c.isalpha()]
    return {
        "avg_word_len": (sum(len(w) for w in words) / nw) if nw else 0.0,
        "caps_ratio": (sum(1 for c in letters if c.isupper()) / len(letters))
                      if letters else 0.0,
        "exclaim_count": text.count("!"),
        "question_count_stat": text.count("?"),
        "emoji_count": len(_EMOJI.findall(text)),
        "bullet_count": len(_BULLET_LINE.findall(text)),
        "placeholder_density": (len(_PLACEHOLDER.findall(text)) / max(nw, 1)) * 100,
    }


def _build_text_stats(posts: pd.DataFrame) -> pd.DataFrame:
    stats = pd.DataFrame([_text_stats(t) for t in posts["text"]],
                         index=posts.index)
    return stats


def _author_cos_l2(emb_df: pd.DataFrame, order_df: pd.DataFrame) -> pd.DataFrame:
    """For each post, cosine and L2 distance between its embedding and the
    mean of the author's PRIOR post embeddings (expanding, shift(1)).
    """
    cols = [c for c in emb_df.columns if c.startswith("e")]
    E = emb_df[cols].values.astype(np.float32)
    posts_order = order_df.reset_index(drop=True)
    # Map (author, post) -> row in E
    key_to_row = pd.Series(np.arange(len(emb_df)),
                           index=pd.MultiIndex.from_frame(emb_df[["author", "post"]]))
    rows = key_to_row.reindex(
        pd.MultiIndex.from_frame(posts_order[["author", "post"]])
    ).values
    # Expanding mean per author in posts_order (which is sorted by author,time)
    cos = np.zeros(len(posts_order), dtype=np.float32)
    l2 = np.zeros(len(posts_order), dtype=np.float32)
    author_count = np.zeros(len(posts_order), dtype=np.int32)

    prev_author = None
    running_sum = None
    running_n = 0
    for i, (idx, auth) in enumerate(zip(rows, posts_order["author"].values)):
        if auth != prev_author:
            running_sum = np.zeros(DIM, dtype=np.float32)
            running_n = 0
            prev_author = auth
        author_count[i] = running_n
        if running_n == 0 or idx < 0:
            cos[i] = 0.0
            l2[i] = 0.0
        else:
            e = E[idx]
            mean = running_sum / running_n
            m_norm = float(np.linalg.norm(mean))
            e_norm = float(np.linalg.norm(e))
            if m_norm > 0 and e_norm > 0:
                cos[i] = float(np.dot(e, mean) / (m_norm * e_norm))
            else:
                cos[i] = 0.0
            l2[i] = float(np.linalg.norm(e - mean))
        if idx >= 0:
            running_sum += E[idx]
            running_n += 1

    out = pd.DataFrame({
        "author": posts_order["author"].values,
        "post": posts_order["post"].values,
        "cos_to_author": cos,
        "l2_to_author": l2,
        "author_posts_so_far": author_count,
    })
    return out


def build(tier1_df: pd.DataFrame, train_authors: set,
          pca_n: int = PCA_N) -> pd.DataFrame:
    """Return a per-post DataFrame with Tier 2 columns, keyed by author+post."""
    if not EMB_PATH.exists():
        raise SystemExit(f"missing {EMB_PATH} — run `python -m my.text_embed`")
    emb_df = pd.read_parquet(EMB_PATH)
    # Order posts chronologically per author (same order as tier1)
    order_df = (tier1_df.sort_values(["author", "posted_at"], kind="mergesort")
                [["author", "post"]].reset_index(drop=True))

    cos_l2 = _author_cos_l2(emb_df, order_df)

    # Text stats
    stats = _build_text_stats(tier1_df[["text"]].reset_index(drop=True))
    stats.insert(0, "post", tier1_df["post"].values)
    stats.insert(0, "author", tier1_df["author"].values)

    # PCA fit on train posts only — but we need the embeddings aligned.
    # Build index for emb_df
    emb_df_idx = emb_df.set_index(["author", "post"])
    tier1_keys = pd.MultiIndex.from_frame(tier1_df[["author", "post"]])
    E_aligned = emb_df_idx.reindex(tier1_keys)
    cols = [c for c in E_aligned.columns if c.startswith("e")]
    E_mat = E_aligned[cols].values.astype(np.float32)
    train_mask = tier1_df["author"].isin(train_authors).values

    from sklearn.decomposition import PCA
    pca = PCA(n_components=pca_n, random_state=42)
    pca.fit(E_mat[train_mask])
    pca_mat = pca.transform(E_mat).astype(np.float32)
    pca_df = pd.DataFrame(pca_mat, columns=PCA_COLS[:pca_n])
    pca_df.insert(0, "post", tier1_df["post"].values)
    pca_df.insert(0, "author", tier1_df["author"].values)

    # Combine
    out = (stats
           .merge(cos_l2, on=["author", "post"], how="left")
           .merge(pca_df, on=["author", "post"], how="left"))
    return out


def materialise(tier1_cache: str = "cache/tier1_features.parquet",
                split_path: str = "data/split.json",
                out_path: str = "cache/tier2_features.parquet",
                force: bool = False) -> pd.DataFrame:
    out_path = Path(out_path)
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)
    import json
    tier1 = pd.read_parquet(tier1_cache)
    split = json.loads(Path(split_path).read_text())
    df = build(tier1, set(split["train"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


if __name__ == "__main__":
    df = materialise(force=True)
    print(f"wrote cache/tier2_features.parquet: {len(df):,} rows, {df.shape[1]} cols")
    print("columns:", list(df.columns)[:10], "...", list(df.columns)[-5:])
