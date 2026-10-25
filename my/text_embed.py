"""Encode every post's text with multilingual-e5-small, cache in parquet.

e5 convention: pass "passage: <text>" for documents to retrieve, "query: ..."
for queries. We use passage for every post. Output is L2-normalized 384-dim.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Make CPU inference deterministic-ish and keep RAM in check
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_num_threads(max(1, (os.cpu_count() or 2)))

MODEL_NAME = "intfloat/multilingual-e5-small"
DIM = 384
PASSAGE_PREFIX = "passage: "
MAX_LEN = 160     # covers ~p85 of posts by token count
BATCH_SIZE = 96


def _load_posts_text(data_dir: str | Path) -> pd.DataFrame:
    """Just author/post/text — pulled from the Tier 1 cache if present
    (avoids re-reading 5433 JSON feeds)."""
    cache = Path("cache/tier1_features.parquet")
    if cache.exists():
        return pd.read_parquet(cache, columns=["author", "post", "text"])
    from kairos.data import load_posts
    posts = load_posts(data_dir)
    return posts[["author", "post", "text"]]


def encode(model, texts: list[str]) -> np.ndarray:
    with torch.inference_mode():
        enc = model.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        out = model(enc)
        emb = out["sentence_embedding"]
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cache/text_embeddings_e5.parquet")
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    args = ap.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    posts = _load_posts_text(data_dir="data")
    print(f"{len(posts):,} posts to encode", flush=True)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model.max_seq_length = MAX_LEN
    model.eval()
    print(f"loaded {MODEL_NAME}  max_seq_length={model.max_seq_length}", flush=True)

    texts = [PASSAGE_PREFIX + (t or "") for t in posts["text"].tolist()]
    t0 = time.time()
    emb = model.encode(
        texts,
        batch_size=args.batch,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    dt = time.time() - t0
    print(f"\nencoded {len(texts):,} in {dt:.0f}s "
          f"({len(texts)/max(dt,1):.1f} posts/s)")

    cols = [f"e{i:03d}" for i in range(DIM)]
    df = pd.DataFrame(emb.astype(np.float32), columns=cols)
    df.insert(0, "post", posts["post"].values)
    df.insert(0, "author", posts["author"].values)
    df.to_parquet(out_path, index=False)
    print(f"wrote {out_path}  ({df.shape[0]:,} rows x {df.shape[1]} cols)")


if __name__ == "__main__":
    main()
