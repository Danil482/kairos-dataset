from collections import Counter
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from kairos.text_features import _HASHTAG_RE
TAG_FEATURE_NAMES = ['graph_tag_count', 'graph_tag_unique_count', 'graph_tag_rarity_mean', 'graph_tag_rarity_max', 'graph_tag_known_fraction', 'graph_tag_entropy', 'graph_pair_count', 'graph_pair_pmi_mean', 'graph_pair_pmi_max', 'graph_pair_known_fraction', 'graph_tag_degree_mean', 'graph_tag_degree_max', 'graph_tag_weighted_degree_mean', 'graph_tag_component_size_max', 'graph_tag_pagerank_mean', 'graph_tag_pagerank_max', 'graph_tag_truncated_count']
GRAPH_NEIGHBOR_K = 10
GRAPH_BLOCK_SIZE = 128
MAX_GRAPH_TAGS = 32

def hashtags(text: object) -> list[str]:
    return [m.group(1).casefold() for m in _HASHTAG_RE.finditer(str(text or ''))]

class TextGraphReference:

    def __init__(self, k: int=GRAPH_NEIGHBOR_K, block_size: int=GRAPH_BLOCK_SIZE, jobs: int=1):
        if min(k, block_size, jobs) < 1:
            raise ValueError('k, block_size and jobs must be positive')
        self.k, self.block_size, self.jobs = (k, block_size, jobs)

    def _centralities(self):
        keys = sorted(self.tag_df)
        self.tag_degree, self.weighted_degree = (Counter(), Counter())
        self.component_size, self.pagerank = ({}, {})
        if not keys:
            return
        mapping = {key: i for i, key in enumerate(keys)}
        left, right, weights = ([], [], [])
        for (a, b), weight in self.pair_df.items():
            self.tag_degree[a] += 1
            self.tag_degree[b] += 1
            self.weighted_degree[a] += weight
            self.weighted_degree[b] += weight
            left.extend((mapping[a], mapping[b]))
            right.extend((mapping[b], mapping[a]))
            weights.extend((weight, weight))
        graph = sparse.csr_matrix((np.asarray(weights, dtype=np.float32), (left, right)), shape=(len(keys), len(keys)))
        _, components = connected_components(graph, directed=False)
        sizes = np.bincount(components)
        self.component_size = {key: int(sizes[components[i]]) for key, i in mapping.items()}
        degree = np.asarray(graph.sum(axis=1)).ravel()
        transition = sparse.diags(1.0 / np.maximum(degree, 1)) @ graph
        rank = np.full(len(keys), 1.0 / len(keys), dtype=np.float64)
        for _ in range(100):
            updated = 0.15 / len(keys) + 0.85 * (transition.T @ rank + rank[degree == 0].sum() / len(keys))
            if np.abs(updated - rank).sum() < 1e-08:
                rank = updated
                break
            rank = updated
        self.pagerank = dict(zip(keys, rank))

    def _tag_features(self, texts: pd.Series) -> pd.DataFrame:
        rows = []
        for text in texts.fillna(''):
            tags = hashtags(text)
            all_unique = sorted(set(tags))
            unique = all_unique[:MAX_GRAPH_TAGS]
            freq = np.asarray([self.tag_df.get(tag, 0) for tag in unique], dtype=float)
            known = freq[freq > 0]
            rarity = np.log1p(self.n_docs / np.maximum(freq, 1))
            pair_scores = []
            for a, b in combinations(unique, 2):
                count = self.pair_df.get((a, b), 0)
                if count:
                    pair_scores.append(max(0.0, float(np.log(count * self.n_docs / (self.tag_df[a] * self.tag_df[b])))))
            probs = known / known.sum() if len(known) else np.asarray([])
            degrees = [self.tag_degree.get(tag, 0) for tag in unique]
            ranks = [self.pagerank.get(tag, 0) for tag in unique]
            n_pairs = len(unique) * (len(unique) - 1) // 2
            rows.append([len(tags), len(all_unique), rarity.mean() if len(rarity) else 0, rarity.max() if len(rarity) else 0, len(known) / max(1, len(unique)), -(probs * np.log(probs)).sum(), len(pair_scores), np.mean(pair_scores) if pair_scores else 0, max(pair_scores, default=0), len(pair_scores) / max(1, n_pairs), np.mean(degrees) if degrees else 0, max(degrees, default=0), np.mean([self.weighted_degree.get(tag, 0) for tag in unique]) if unique else 0, max((self.component_size.get(tag, 0) for tag in unique), default=0), np.mean(ranks) if ranks else 0, max(ranks, default=0), len(all_unique) - len(unique)])
        return pd.DataFrame(rows, index=texts.index, columns=TAG_FEATURE_NAMES, dtype='float32')
