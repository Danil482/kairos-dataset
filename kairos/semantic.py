"""
Text-meaning features: similarity to the author's earlier posts, relevance to
the author's listed skills, and kNN statistics over past posts (reactions of
similar earlier posts).

Everything is label-free and uses only post text, profile skills and historical
reaction counts. Similarity and kNN features look strictly backwards in time:
a post is only compared with the author's *earlier* posts.

Speed: one embedding pass over all posts (cached on disk), then vectorised
numpy / sparse algebra. With the default `model2vec` backend the embedding step
takes seconds to a couple of minutes on CPU; the rest is a few seconds.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from kairos.features import SEMANTIC_NUM

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
WINDOWS = (5, 20)  # "previous N posts"; names in SEMANTIC_NUM must match
KNN_KS = (3, 10)  # k for kNN over past posts
MAX_CHARS = 1000  # longer posts are truncated before embedding (speed)
BM25_K1, BM25_B = 1.5, 0.75

MODEL2VEC_MODEL = "minishlab/potion-multilingual-128M"
SBERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def _encode(texts: list[str], backend: str) -> np.ndarray:
    if backend == "model2vec":
        from model2vec import StaticModel

        return StaticModel.from_pretrained(MODEL2VEC_MODEL).encode(texts)
    if backend == "sbert":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(SBERT_MODEL)
        model.max_seq_length = 128
        return model.encode(texts, batch_size=128, show_progress_bar=True)
    if backend == "tfidf":
        tfidf = TfidfVectorizer(strip_accents="unicode", sublinear_tf=True, min_df=2)
        matrix = tfidf.fit_transform(texts)
        return TruncatedSVD(
            min(128, matrix.shape[1] - 1), random_state=0
        ).fit_transform(matrix)
    raise ValueError(f"unknown backend {backend!r} (use model2vec, sbert or tfidf)")


def embed(texts: list[str], backend: str = "model2vec", cache_dir=None) -> np.ndarray:
    """L2-normalised embeddings, one row per text; empty texts get an all-zero row."""
    texts = [t[:MAX_CHARS] for t in texts]

    cache_file = None
    if cache_dir is not None:
        digest = hashlib.sha1("\x00".join([backend, *texts]).encode()).hexdigest()[:16]
        cache_file = Path(cache_dir) / f"embeddings_{backend}_{digest}.npy"
        if cache_file.is_file():
            return np.load(cache_file)

    vectors = np.asarray(_encode(texts, backend), dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    vectors[[not t.strip() for t in texts]] = 0.0

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, vectors)
    return vectors


# --------------------------------------------------------------------------- #
# Author slices
# --------------------------------------------------------------------------- #
def _author_slices(author_codes: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) row ranges of each author in a frame sorted by author."""
    edges = np.flatnonzero(np.diff(author_codes)) + 1
    starts = np.r_[0, edges]
    return list(zip(starts, np.r_[edges, len(author_codes)]))


# --------------------------------------------------------------------------- #
# Pure similarity to earlier posts (no outcomes)
# --------------------------------------------------------------------------- #
def _lag_similarities(
    emb: np.ndarray, valid: np.ndarray, authors: np.ndarray, max_lag: int
):
    """sims[i, k-1] = cosine(post i, the author's k-th previous post); NaN if none."""
    sims = np.full((len(emb), max_lag), np.nan, dtype=np.float32)
    for lag in range(1, min(max_lag, len(emb) - 1) + 1):
        cosine = np.einsum("ij,ij->i", emb[lag:], emb[:-lag])
        usable = (authors[lag:] == authors[:-lag]) & valid[lag:] & valid[:-lag]
        sims[lag:, lag - 1] = np.where(usable, cosine, np.nan)
    return sims


def _nan_mean_max(sims: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise mean and max ignoring NaN; NaN when a row has no values."""
    seen = ~np.isnan(sims)
    count = seen.sum(axis=1)
    mean = np.where(count > 0, np.nansum(sims, axis=1) / np.maximum(count, 1), np.nan)
    peak = np.where(count > 0, np.where(seen, sims, -np.inf).max(axis=1), np.nan)
    return mean, peak


def _centroid_similarities(emb, valid, slices, windows) -> dict[str, np.ndarray]:
    """Cosine between each post and the centroid of the author's earlier posts."""
    out = {w: np.full(len(emb), np.nan, dtype=np.float32) for w in windows}
    for start, end in slices:
        block = emb[start:end].astype(np.float64)
        running = np.vstack([np.zeros((1, block.shape[1])), np.cumsum(block, axis=0)])
        position = np.arange(end - start)
        for w in windows:
            first = 0 if w is None else np.maximum(position - w, 0)
            prior = running[position] - running[first]
            norm = np.linalg.norm(prior, axis=1)
            cosine = (block * prior).sum(axis=1) / np.maximum(norm, 1e-9)
            out[w][start:end] = np.where(
                (norm > 1e-9) & valid[start:end], cosine, np.nan
            )
    return out


def _history_features(emb, valid, author_codes) -> dict[str, np.ndarray]:
    features = {}
    sims = _lag_similarities(emb, valid, author_codes, max(WINDOWS))
    features["sim_prev_1"] = sims[:, 0]
    for w in WINDOWS:
        features[f"sim_mean_{w}"], features[f"sim_max_{w}"] = _nan_mean_max(sims[:, :w])

    centroid = _centroid_similarities(
        emb, valid, _author_slices(author_codes), [WINDOWS[-1], None]
    )
    features[f"sim_centroid_{WINDOWS[-1]}"] = centroid[WINDOWS[-1]]
    features["sim_centroid_all"] = centroid[None]
    return features


# --------------------------------------------------------------------------- #
# kNN over past posts using embedding cosine similarity
# --------------------------------------------------------------------------- #
def _embedding_knn_features(emb, valid, author_codes, reactions, ks=KNN_KS):
    n = len(emb)
    out = {}
    for k in ks:
        out[f"knn_emb_mean_reaction_{k}"] = np.full(n, np.nan, dtype=np.float32)
        out[f"knn_emb_max_reaction_{k}"] = np.full(n, np.nan, dtype=np.float32)
        out[f"knn_emb_min_reaction_{k}"] = np.full(n, np.nan, dtype=np.float32)

    for start, end in _author_slices(author_codes):
        block = emb[start:end]
        block_valid = valid[start:end]
        m = len(block)
        if m <= 1:
            continue
        sim = block @ block.T  # (m, m) cosine, block rows L2-normalised

        for i in range(m):
            if not block_valid[i]:
                continue
            earlier_valid = block_valid[:i]
            if not earlier_valid.any():
                continue
            sims_i = sim[i, :i].copy()
            sims_i[~earlier_valid] = -np.inf
            order = np.argsort(-sims_i)  # best first; -inf sort last
            order = order[np.isfinite(sims_i[order])]
            if len(order) == 0:
                continue

            for k in ks:  # k bound here, per-write
                top = order[:k]
                nr = reactions[start + top]
                nr = nr[np.isfinite(nr)]
                if len(nr) == 0:
                    continue
                out[f"knn_emb_mean_reaction_{k}"][start + i] = nr.mean()
                out[f"knn_emb_max_reaction_{k}"][start + i] = nr.max()
                out[f"knn_emb_min_reaction_{k}"][start + i] = nr.min()

    return out


# --------------------------------------------------------------------------- #
# kNN over past posts using BM25 lexical similarity
# --------------------------------------------------------------------------- #
def _bm25_knn_features(
    texts: list[str],
    author_codes: np.ndarray,
    reactions: np.ndarray,
    ks: tuple[int, ...] = KNN_KS,
) -> dict[str, np.ndarray]:
    """
    kNN over past posts using BM25 lexical similarity.

    For each post, score it against the author's *earlier* posts using BM25
    (current post is the query, earlier post is the document), take the top-k,
    and aggregate those neighbours' reactions. Strictly backward-looking.

    `reactions` is aligned with `texts` (NaN where unknown). Neighbours with
    NaN reactions are dropped from the aggregate; a post with no usable
    neighbours gets NaN.
    """
    n = len(texts)
    out: dict[str, np.ndarray] = {}
    for k in ks:
        out[f"knn_bm25_mean_reaction_{k}"] = np.full(n, np.nan, dtype=np.float32)
        out[f"knn_bm25_max_reaction_{k}"] = np.full(n, np.nan, dtype=np.float32)
        out[f"knn_bm25_min_reaction_{k}"] = np.full(n, np.nan, dtype=np.float32)

    # ---- BM25 precompute (global vocabulary, per-document term weights) -----
    vectorizer = CountVectorizer(strip_accents="unicode", min_df=1)
    counts = vectorizer.fit_transform(texts).tocsr().astype(np.float32)
    n_docs = counts.shape[0]

    doc_freq = np.bincount(counts.indices, minlength=counts.shape[1])
    idf = np.log1p((n_docs - doc_freq + 0.5) / (doc_freq + 0.5))

    length = np.asarray(counts.sum(axis=1)).ravel()
    avg_length = max(float(length.mean()), 1.0)

    rows = np.repeat(np.arange(n_docs), np.diff(counts.indptr))
    norm = 1 - BM25_B + BM25_B * length[rows] / avg_length
    tf = counts.data
    counts.data = (
        tf * (BM25_K1 + 1) / (tf + BM25_K1 * norm) * idf[counts.indices]
    ).astype(np.float32)
    bm25_matrix = counts  # row i holds BM25 term weights of document i

    # ---- kNN per author, strictly earlier posts only -----------------------
    for start, end in _author_slices(author_codes):
        m = end - start
        if m <= 1:
            continue
        author_bm25 = bm25_matrix[start:end]  # sparse (m, vocab)

        for i in range(1, m):
            # score current post (i) as query against earlier posts (0..i-1) as docs
            scores = np.asarray(
                (author_bm25[:i] @ author_bm25[i].T).todense()
            ).ravel()  # shape (i,)
            if scores.size == 0 or scores.max() <= 0:
                continue

            order = np.argsort(-scores)  # best first

            for k in ks:
                top = order[:k]
                nr = reactions[start + top]
                nr = nr[np.isfinite(nr)]
                if nr.size == 0:
                    continue
                out[f"knn_bm25_mean_reaction_{k}"][start + i] = nr.mean()
                out[f"knn_bm25_max_reaction_{k}"][start + i] = nr.max()
                out[f"knn_bm25_min_reaction_{k}"][start + i] = nr.min()

    return out


# --------------------------------------------------------------------------- #
# Skill relevance (BM25 + embedding)
# --------------------------------------------------------------------------- #
def _bm25_features(texts: list[str], author_codes: np.ndarray, skill_docs: list[str]):
    """BM25 with the author's skills as the query and every post as a document."""
    vectorizer = CountVectorizer(strip_accents="unicode")
    vectorizer.fit([*texts, *skill_docs])
    counts = vectorizer.transform(texts).tocsr().astype(np.float32)
    query = (vectorizer.transform(skill_docs) > 0).astype(np.float32).tocsr()

    n_docs = counts.shape[0]
    doc_freq = np.bincount(counts.indices, minlength=counts.shape[1])
    idf = np.log1p((n_docs - doc_freq + 0.5) / (doc_freq + 0.5))
    length = np.asarray(counts.sum(axis=1)).ravel()
    rows = np.repeat(np.arange(n_docs), np.diff(counts.indptr))
    norm = 1 - BM25_B + BM25_B * length[rows] / max(length.mean(), 1.0)
    tf = counts.data
    counts.data = (
        tf * (BM25_K1 + 1) / (tf + BM25_K1 * norm) * idf[counts.indices]
    ).astype(np.float32)

    matched = counts.multiply(query[author_codes]).tocsr()
    score = np.asarray(matched.sum(axis=1)).ravel()
    hits = np.diff(matched.indptr)
    n_terms = np.asarray(query.sum(axis=1)).ravel()[author_codes]

    known = n_terms > 0
    per_term = np.where(known, score / np.maximum(n_terms, 1), np.nan)
    hit_share = np.where(known, hits / np.maximum(n_terms, 1), np.nan)
    return {
        "skill_bm25": np.where(known, score, np.nan),
        "skill_bm25_per_term": per_term,
        "skill_hit_share": hit_share,
    }


def _skill_embedding_features(
    emb, valid, slices, skill_vectors
) -> dict[str, np.ndarray]:
    """Cosine of each post to its author's closest skill and to their skill centroid."""
    best = np.full(len(emb), np.nan, dtype=np.float32)
    centre = np.full(len(emb), np.nan, dtype=np.float32)
    for (start, end), skills in zip(slices, skill_vectors):
        if len(skills) == 0:
            continue
        posts = emb[start:end]
        mean = skills.mean(axis=0)
        mean /= max(np.linalg.norm(mean), 1e-9)
        ok = valid[start:end]
        best[start:end] = np.where(ok, (posts @ skills.T).max(axis=1), np.nan)
        centre[start:end] = np.where(ok, posts @ mean, np.nan)
    return {"skill_emb_max": best, "skill_emb_centroid": centre}


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def build_semantic_features(
    posts: pd.DataFrame,
    skills: dict[str, list[str]],
    backend: str = "model2vec",
    cache_dir=None,
    reactions: str | None = "log_reactions",
) -> pd.DataFrame:
    """
    Return one row per post: ``author``, ``post`` and the ``SEMANTIC_NUM`` columns.

    Parameters
    ----------
    posts:
        Needs ``author``, ``post``, ``posted_at`` and ``text`` (e.g. ``load_posts()``).
    skills:
        Author -> list of skill names (``data.load_author_skills``).
    backend:
        ``"model2vec"`` (fast, multilingual; default), ``"sbert"`` (slower, GPU
        recommended) or ``"tfidf"`` (offline fallback).
    cache_dir:
        Where to store embeddings so reruns skip the embedding step.
    reactions:
        Column name in `posts` holding a numeric reaction summary (e.g.
        ``"log_reactions"``). If ``None`` or missing, kNN features are skipped.
    """
    frame = (
        posts[
            ["author", "post", "posted_at", "text", *([reactions] if reactions else [])]
        ]
        .sort_values(["author", "posted_at"], kind="mergesort")
        .reset_index(drop=True)
    )
    frame["text"] = frame["text"].fillna("").astype(str)
    author_codes = pd.factorize(frame["author"])[0]
    authors = frame["author"].drop_duplicates().tolist()  # in code order
    slices = _author_slices(author_codes)

    # One embedding call for posts and the (few, short) unique skill names.
    skill_names = sorted({s for a in authors for s in skills.get(a, [])})
    vectors = embed([*frame["text"], *skill_names], backend, cache_dir)
    emb, skill_emb = vectors[: len(frame)], vectors[len(frame) :]
    valid = np.linalg.norm(emb, axis=1) > 0

    skill_row = {name: i for i, name in enumerate(skill_names)}
    skill_vectors = [
        (
            skill_emb[[skill_row[s] for s in skills.get(a, [])]]
            if skills.get(a)
            else np.empty((0, emb.shape[1]), dtype=np.float32)
        )
        for a in authors
    ]
    skill_docs = [" ".join(skills.get(a, [])) for a in authors]

    # Historical reaction signal for kNN features
    if reactions and reactions in frame.columns:
        reaction_values = frame[reactions].to_numpy(dtype=np.float32)
        # replace inf/-inf with nan
        reaction_values = np.where(
            np.isfinite(reaction_values), reaction_values, np.nan
        )
    else:
        reaction_values = np.full(len(frame), np.nan, dtype=np.float32)

    columns = {
        **_history_features(emb, valid, author_codes),
        **_bm25_features(list(frame["text"]), author_codes, skill_docs),
        **_skill_embedding_features(emb, valid, slices, skill_vectors),
    }

    # kNN features only if we have at least some finite reactions
    if np.isfinite(reaction_values).any():
        columns.update(
            _embedding_knn_features(
                emb, valid, author_codes, reaction_values, ks=KNN_KS
            )
        )
        columns.update(
            _bm25_knn_features(
                list(frame["text"]), author_codes, reaction_values, ks=KNN_KS
            )
        )

    out = pd.concat([frame[["author", "post"]], pd.DataFrame(columns)], axis=1)
    if set(out.columns) - {"author", "post"} != set(SEMANTIC_NUM):
        raise RuntimeError(
            "semantic columns differ from features.SEMANTIC_NUM; "
            "update SEMANTIC_NUM to include the new kNN columns"
        )
    return out[["author", "post", *SEMANTIC_NUM]]
