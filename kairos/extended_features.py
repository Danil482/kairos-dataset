from collections import deque
import numpy as np
import pandas as pd

def semantic_reaction_history(ids, vectors, posts, history_ids=None):
    source = posts.set_index(['author', 'post'])
    history_ids = ids if history_ids is None else history_ids
    keys = pd.MultiIndex.from_frame(history_ids[['author', 'post']])
    current = source.reindex(keys).reset_index()
    if len(vectors) != len(current) or current.posted_at.isna().any():
        raise ValueError('History vectors and source rows differ')
    output = np.zeros((len(current), 9), np.float32)
    for _, group in current.groupby('author', sort=False):
        history = deque(maxlen=30)
        for stamp, tied in group.sort_values('posted_at').groupby('posted_at', sort=True):
            rows = tied.index.to_numpy()
            for row in rows:
                if not history:
                    continue
                indices = np.asarray(history)
                past = current.iloc[indices]
                reactions = np.log1p(past.reactions.to_numpy(float))
                similarity = np.clip(vectors[indices] @ vectors[row], 0, 1)
                weights = np.exp((similarity - similarity.max()) / 0.1)
                level = float(np.median(reactions))
                weighted = float(weights @ reactions / weights.sum())
                top = np.argsort(similarity)[-min(5, len(indices)):]
                ages = np.maximum((pd.to_datetime(past.crawled_at) - pd.to_datetime(past.posted_at)).dt.total_seconds().to_numpy() / 86400, 1.0)
                available = np.isfinite(ages)
                rates = reactions[available] - 0.25 * np.log1p(ages[available])
                age_delta = reactions[-1] - 0.25 * np.log1p(ages[-1]) - np.median(rates) if available[-1] and len(rates) else 0.0
                output[row] = [weighted - level, np.mean(reactions[top]) - level, similarity.max(), similarity.mean(), np.sqrt(weights @ (reactions - weighted) ** 2 / weights.sum()), float(len(indices)), age_delta, np.mean(reactions[-5:]) - np.mean(reactions), 1 - float(available.mean())]
            history.extend(rows.tolist())
    names = ['semantic_reaction_weighted_delta', 'semantic_reaction_top5_delta', 'semantic_reaction_similarity_max', 'semantic_reaction_similarity_mean', 'semantic_reaction_weighted_std', 'semantic_reaction_prior_count', 'semantic_reaction_age_adjusted_last_delta', 'semantic_reaction_recent_delta', 'semantic_reaction_age_missing_fraction']
    select = keys.get_indexer(pd.MultiIndex.from_frame(ids[['author', 'post']]))
    if (select < 0).any():
        raise ValueError('Requested post missing from semantic history')
    return pd.DataFrame(output[select], columns=names)
