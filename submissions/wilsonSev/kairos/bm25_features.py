"""BM25 retrieval of strictly earlier own posts (Lecture 3, slides 14 and 45)."""

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Same default word tokenization as our TF-IDF experiment; no fitted vocabulary.
_tokenize = CountVectorizer().build_analyzer()


def _scores(query, index, lengths, k1, b):
    """Score a query against an index containing ONLY earlier documents."""
    n = len(lengths)
    scores = np.zeros(n)
    if not n or not sum(lengths):
        return scores
    lengths = np.asarray(lengths, dtype=float)
    length_penalty = k1 * (1 - b + b * lengths / lengths.mean())
    for word in sorted(set(query)):  # Each query term contributes once.
        matches = index.get(word, [])
        if not matches:
            continue
        df = len(matches)  # Number of history documents containing this word.
        idf = np.log1p((n - df + 0.5) / (df + 0.5))
        positions, frequencies = np.asarray(matches).T
        scores[positions] += (
            idf * frequencies * (k1 + 1) / (frequencies + length_penalty[positions])
        )
    return scores


def _add_document(index, counts, position):
    for word, frequency in counts.items():
        index[word].append((position, frequency))


def bm25_scores(query, documents, k1=1.2, b=0.75):
    """Small inspectable example: one query, one score per supplied document."""
    if not np.isfinite(k1) or k1 <= 0 or not 0 <= b <= 1:
        raise ValueError("Require finite k1 > 0 and 0 <= b <= 1")
    index, lengths = defaultdict(list), []
    for position, text in enumerate(documents):
        counts = Counter(_tokenize(text or ""))
        lengths.append(sum(counts.values()))
        _add_document(index, counts, position)
    return _scores(_tokenize(query or ""), index, lengths, k1, b)


def author_bm25_features(posts, top_k=3, k1=1.2, b=0.75):
    """Retrieve earlier posts, then summarize their observed engagement.

    Corpus = the same author's strictly earlier own posts. Recompute IDF and
    average length from that history, including empty documents. Query terms
    are unique; lowercase word tokens have >=2 characters, without stopword
    removal/stemming. Select up to top_k positive scores; ties prefer earlier
    time then post ID. Equal-time posts cannot enter each other's history.

    bm25_history_delta = mean log1p(reactions) of matches minus median
    log1p(reactions) of ALL earlier own posts. Zero is a fallback when there
    are no matches, distinguished by bm25_match_count=0. Unknown-date rows
    return this fallback and never enter history. Top IDs/scores are diagnostics.
    Current/future reactions and labels are never used for a row's features.
    Dataset counters are crawl snapshots, not historical as-of counters.
    """
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not np.isfinite(k1) or k1 <= 0 or not 0 <= b <= 1:
        raise ValueError("Require finite k1 > 0 and 0 <= b <= 1")
    table = posts[["author", "post", "posted_at", "text", "reactions"]].reset_index(
        drop=True
    )
    if (
        table[["author", "post"]].isna().any().any()
        or table.duplicated(["author", "post"]).any()
    ):
        raise ValueError("Expected present, unique author/post identifiers")
    table["text"] = table["text"].fillna("")
    reactions = table["reactions"].to_numpy(dtype=float)
    if not np.isfinite(reactions).all():
        raise ValueError(
            "Reactions must be finite; use the dataset loader's zero convention"
        )
    log_reactions = np.log1p(np.maximum(reactions, 0))
    deltas = np.zeros(len(table))
    counts = np.zeros(len(table), dtype=int)
    top_posts = [[] for _ in range(len(table))]
    top_scores = [[] for _ in range(len(table))]

    for _, author_posts in table.groupby("author", sort=False):
        author_posts = author_posts.sort_values(["posted_at", "post"])
        index, lengths, history_rows = defaultdict(list), [], []
        # NaT timestamps are excluded by groupby.
        for _, batch in author_posts.groupby("posted_at", sort=True):
            usual = np.median(log_reactions[history_rows]) if history_rows else 0.0
            for row in batch.index:
                query = _tokenize(table.at[row, "text"] or "")
                scores = _scores(query, index, lengths, k1, b)
                positive = np.flatnonzero(scores > 0)
                selected = positive[
                    np.argsort(-scores[positive], kind="stable")[:top_k]
                ]
                if len(selected):
                    matched_rows = np.asarray(history_rows)[selected]
                    deltas[row] = log_reactions[matched_rows].mean() - usual
                    counts[row] = len(selected)
                    top_posts[row] = table.loc[matched_rows, "post"].tolist()
                    top_scores[row] = scores[selected].tolist()
            # Update history after scoring the entire timestamp batch.
            for row in batch.index:
                words = Counter(_tokenize(table.at[row, "text"] or ""))
                _add_document(index, words, len(history_rows))
                lengths.append(sum(words.values()))
                history_rows.append(row)

    return table[["author", "post"]].assign(
        bm25_history_delta=deltas,
        bm25_match_count=counts,
        bm25_top_posts=top_posts,
        bm25_top_scores=top_scores,
    )
