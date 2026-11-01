"""Text similarity to each author's strictly earlier posts (Lecture 3, 32/45)."""

import numpy as np
import pandas as pd


def author_text_similarity(posts: pd.DataFrame, vectorizer) -> pd.DataFrame:
    """Transform with an already fitted TF-IDF vectorizer; never refit it.

    Return one row per (author, post), in input order. Only strictly earlier
    timestamps from the same author enter the mean profile. Equal-time posts
    are scored together before adding any of them to history. Unknown-date
    posts score zero and are excluded from all histories. Empty histories,
    zero profiles and zero current vectors also score zero. History counts
    are diagnostic only; the experiment adds just history_text_similarity.
    No labels, reactions, or engagement fields are read.
    """
    table = posts[["author", "post", "posted_at", "text"]].reset_index(drop=True)
    if table[["author", "post"]].isna().any().any():
        raise ValueError("Author and post identifiers must be present")
    if table.duplicated(["author", "post"]).any():
        raise ValueError("Expected a unique row per author and post")
    vectors = vectorizer.transform(table["text"].fillna("")).tocsr()
    similarities = np.zeros(len(table))
    history_counts = np.zeros(len(table), dtype=int)

    for _, author_posts in table.groupby("author", sort=False):
        history_sum = np.zeros(vectors.shape[1])
        history_size = 0
        # groupby excludes NaT and sorts the remaining timestamps.
        for _, same_time_posts in author_posts.groupby("posted_at", sort=True):
            rows = same_time_posts.index.to_numpy()
            current_vectors = vectors[rows]
            history_counts[rows] = history_size
            if history_size:
                profile = history_sum / history_size
                profile_norm = np.linalg.norm(profile)
                row_norms = np.sqrt(
                    current_vectors.multiply(current_vectors).sum(axis=1)
                ).A1
                denominator = row_norms * profile_norm
                similarities[rows] = np.divide(
                    current_vectors @ profile,
                    denominator,
                    out=np.zeros(len(rows)),
                    where=denominator > 0,
                )
            # Update only after every post at this timestamp has been scored.
            history_sum += np.asarray(current_vectors.sum(axis=0)).ravel()
            history_size += len(rows)

    return table[["author", "post"]].assign(
        history_text_similarity=np.clip(similarities, 0.0, 1.0),
        text_history_n=history_counts,
    )
