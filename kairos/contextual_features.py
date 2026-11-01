from collections import deque
import numpy as np
import pandas as pd

def semantic_history(ids, embeddings, timestamps, windows=(5, 20)):
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or len(ids) != len(vectors) or len(timestamps) != len(ids):
        raise ValueError('Unaligned context inputs')
    if not np.isfinite(vectors).all() or ids[['author', 'post']].duplicated().any():
        raise ValueError('Invalid vectors or duplicate post keys')
    table = ids[['author', 'post']].copy().reset_index(drop=True)
    table['time'] = pd.to_datetime(np.asarray(timestamps), utc=True, errors='coerce')
    table['row'] = np.arange(len(table))
    width = vectors.shape[1]
    block_width = 2 * width + 5
    result = np.zeros((len(table), len(windows) * block_width), np.float32)
    for _, group in table.dropna(subset=['time']).groupby('author', sort=False):
        past = deque(maxlen=max(windows))
        for _, tied in group.sort_values('time').groupby('time', sort=True):
            rows = tied['row'].to_numpy()
            for row in rows:
                current = vectors[row]
                for index, window in enumerate(windows):
                    history = list(past)[-window:]
                    offset = index * block_width
                    if not history:
                        continue
                    previous = vectors[history]
                    mean = previous.mean(axis=0)
                    difference = current - mean
                    mean_norm = np.linalg.norm(mean)
                    norm = np.linalg.norm(current)
                    similarity = float(current @ mean / max(norm * mean_norm, 1e-08))
                    sims = previous @ current / np.maximum(np.linalg.norm(previous, axis=1) * norm, 1e-08)
                    result[row, offset:offset + width] = mean
                    result[row, offset + width:offset + 2 * width] = difference
                    result[row, offset + 2 * width:offset + block_width] = (len(history), similarity, sims.max(), sims.std(), mean_norm)
            past.extend(rows.tolist())
    names = []
    for window in windows:
        names += [f'prior{window}_mean_{i}' for i in range(width)]
        names += [f'prior{window}_difference_{i}' for i in range(width)]
        names += [f'prior{window}_{name}' for name in ('count', 'cosine', 'max_cosine', 'cosine_std', 'mean_norm')]
    return (result, names)
