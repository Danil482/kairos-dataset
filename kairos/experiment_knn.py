"""Exact cross-author cosine neighbors from an explicitly permitted label bank.

The caller owns outer/inner fold construction. This module never extends the
bank, reads query labels, or fetches targets from another data source.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["author", "post"]
KNN_COLUMNS = [
    "knn_fraction_0", "knn_fraction_1", "knn_fraction_2", "knn_mean_residual",
    "knn_mean_cosine", "knn_max_cosine", "knn_mean_age_days",
    "knn_neighbor_count", "knn_missing",
]
_DAY_NS = 86_400_000_000_000
_MAX_SCORE_BYTES = 32 * 1024 * 1024


def _keys(frame: pd.DataFrame, name: str) -> pd.MultiIndex:
    if not set(KEYS) <= set(frame):
        raise ValueError(f"{name} needs author and post keys")
    if frame[KEYS].isna().any().any() or frame[KEYS].astype(str).eq("").any().any():
        raise ValueError(f"{name} contains missing or empty post keys")
    index = pd.MultiIndex.from_frame(frame[KEYS])
    if index.has_duplicates:
        raise ValueError(f"{name} contains duplicate post keys")
    return index


def _empty_result(queries: pd.DataFrame) -> pd.DataFrame:
    result = queries[KEYS].reset_index(drop=True).copy()
    for name in KNN_COLUMNS[:7]:
        result[name] = np.nan
    result["knn_neighbor_count"] = np.zeros(len(result), dtype=np.int32)
    result["knn_missing"] = np.ones(len(result), dtype=np.int8)
    return result


def knn_features(history: pd.DataFrame, vectors: np.ndarray, queries: pd.DataFrame,
                 bank: pd.DataFrame, k: int = 20, window_days: float | None = None,
                 device: str = "cpu", batch_size: int = 128) -> pd.DataFrame:
    """Compute target summaries of earlier posts by other authors.

    History supplies author/post/posted_at, content_text (or text), log_reactions
    and level. Vectors are aligned to its row order. Queries supply keys only;
    extra columns, including labels, are ignored. Bank supplies explicit keys
    and label in {0,1,2}. Its residual comes from the matching history row's
    log_reactions - level, not from a current query's outcome.

    Candidates must have nonempty text and a finite nonzero vector, belong to a
    different author, and be strictly earlier than the query. An optional window
    includes ages <= window_days. Keep each author's closest post first, then
    select k authors. Exact cosine ties use ascending (author, post) keys.

    Class fractions and mean residual use weights max(cosine, 0). Mean cosine
    and age are ordinary means over selected neighbors. If all weights are zero,
    target summaries are NaN and missing=1, but the selected count/cosine/age
    remain informative. With no eligible neighbors, numeric values are NaN,
    count=0, missing=1. Missing residuals are omitted only from the residual mean.

    Similarities use float32 in query blocks (at most 128 rows and approximately
    32 MiB each), never a full dataset-by-dataset matrix. CUDA is optional and
    only accelerates matrix multiplication; filtering/aggregation are identical.
    """
    for name, value in (("k", k), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if window_days is not None and (not np.isfinite(window_days) or window_days <= 0):
        raise ValueError("window_days must be positive and finite, or None")
    if device != "cpu" and not (device == "cuda" or device.startswith("cuda:")):
        raise ValueError("device must be cpu or a CUDA device")

    history_index = _keys(history, "history")
    query_index = _keys(queries, "queries")
    query_positions = history_index.get_indexer(query_index)
    if (query_positions < 0).any():
        raise ValueError("A query is absent from history")
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != len(history) or values.shape[1] == 0:
        raise ValueError("vectors must be 2D and aligned with history rows")
    output = _empty_result(queries)
    if queries.empty or bank.empty:
        return output
    text_column = "content_text" if "content_text" in history else "text"
    required = {text_column, "posted_at", "log_reactions", "level"}
    if not required <= set(history):
        raise ValueError(f"history is missing columns: {sorted(required - set(history))}")
    bank_index = _keys(bank, "bank")
    if "label" not in bank or not bank.label.isin([0, 1, 2]).all():
        raise ValueError("bank must contain labels 0, 1, 2")
    bank_positions = history_index.get_indexer(bank_index)
    if (bank_positions < 0).any():
        raise ValueError("A bank post is absent from history")
    dates = pd.to_datetime(history["posted_at"], utc=True, errors="raise")
    if dates.isna().any():
        raise ValueError("history timestamps must not be missing")
    times = dates.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    texts = history[text_column].fillna("").astype(str).str.strip()
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    valid = ((texts.ne("") & texts.ne("__empty__")).to_numpy()
             & np.isfinite(values).all(axis=1) & (norms > 0))
    unit = np.zeros_like(values)
    unit[valid] = values[valid] / norms[valid, None]

    permitted = bank[KEYS + ["label"]].copy().reset_index(drop=True)
    permitted["_position"] = bank_positions
    permitted = permitted.loc[valid[bank_positions]].sort_values(KEYS, kind="mergesort")
    if permitted.empty:
        return output
    positions = permitted["_position"].to_numpy(dtype=int)
    bank_vectors = np.ascontiguousarray(unit[positions])
    bank_times = times[positions]
    bank_authors = permitted.author.to_numpy()
    labels = permitted.label.to_numpy(dtype=int)
    allowed_history = history.iloc[positions]
    log_reactions = pd.to_numeric(allowed_history.log_reactions, errors="raise").to_numpy(dtype=float)
    levels = pd.to_numeric(allowed_history.level, errors="raise").to_numpy(dtype=float)
    residuals = log_reactions - levels
    starts = np.r_[0, np.flatnonzero(bank_authors[1:] != bank_authors[:-1]) + 1]
    ends = np.r_[starts[1:], len(permitted)]
    query_authors = queries.author.to_numpy()
    take = min(k, len(starts))
    block_size = min(batch_size, 128, max(1, _MAX_SCORE_BYTES // (4 * len(permitted))))
    # For even a small dataset, do not materialize all query rows at once.
    if len(queries) > 1:
        block_size = min(block_size, max(1, len(queries) // 2))

    torch = gpu_bank = previous_tf32 = None
    if device != "cpu":
        try:
            import torch as torch_module
        except ImportError as exc:
            raise RuntimeError("CUDA KNN requires PyTorch") from exc
        torch = torch_module
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA KNN requested but CUDA is unavailable")
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    result = output[KNN_COLUMNS].to_numpy(dtype=float)
    try:
        if torch is not None:
            torch.backends.cuda.matmul.allow_tf32 = False
            gpu_bank = torch.as_tensor(bank_vectors, device=device)
        for first in range(0, len(queries), block_size):
            last = min(first + block_size, len(queries))
            query_rows = query_positions[first:last]
            if torch is None:
                similarities = unit[query_rows] @ bank_vectors.T
            else:
                with torch.inference_mode():
                    gpu_queries = torch.as_tensor(unit[query_rows], device=device)
                    similarities = (gpu_queries @ gpu_bank.T).cpu().numpy()
            np.clip(similarities, -1, 1, out=similarities)
            eligible = ((bank_times[None, :] < times[query_rows, None])
                        & (bank_authors[None, :] != query_authors[first:last, None])
                        & valid[query_rows, None])
            if window_days is not None:
                eligible &= bank_times[None, :] >= times[query_rows, None] - window_days * _DAY_NS
            similarities[~eligible] = -np.inf
            author_best = np.maximum.reduceat(similarities, starts, axis=1)
            # Authors are lexicographically sorted; stable ties therefore follow KEY.
            top_authors = np.argsort(-author_best, axis=1, kind="stable")[:, :take]
            for local, author_choices in enumerate(top_authors):
                author_choices = author_choices[np.isfinite(author_best[local, author_choices])]
                if not len(author_choices):
                    continue
                selected = np.asarray([
                    starts[author] + np.argmax(similarities[local, starts[author]:ends[author]])
                    for author in author_choices
                ], dtype=int)
                scores = similarities[local, selected].astype(float)
                weights = np.maximum(scores, 0)
                row = first + local
                result[row, 4] = scores.mean()
                result[row, 5] = scores.max()
                result[row, 6] = ((times[query_rows[local]] - bank_times[selected]) / _DAY_NS).mean()
                result[row, 7] = len(selected)
                total = weights.sum()
                if total <= 0:
                    continue
                result[row, :3] = np.bincount(labels[selected], weights=weights, minlength=3) / total
                known = np.isfinite(residuals[selected]) & (weights > 0)
                if known.any():
                    result[row, 3] = np.average(residuals[selected][known], weights=weights[known])
                result[row, 8] = 0
    finally:
        if torch is not None:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    output[KNN_COLUMNS] = result
    output["knn_neighbor_count"] = output["knn_neighbor_count"].astype("int32")
    output["knn_missing"] = output["knn_missing"].astype("int8")
    return output
