from pathlib import Path
from collections import Counter
import json
import re
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
NETWORK_NAMES = ['reference_count', 'reference_known_fraction', 'reference_unique_authors_mean', 'reference_unique_authors_max', 'reference_post_count_mean', 'reference_post_count_max', 'reference_pagerank_mean', 'reference_pagerank_max', 'reference_component_max', 'reference_degree_mean', 'reference_degree_max', 'reference_owner_incoming', 'reference_owner_pagerank', 'reference_repeated_author_fraction', 'reference_is_quote', 'reference_has_org', 'reference_has_author']

def global_code(value):
    return isinstance(value, str) and bool(re.fullmatch('(?:AUTHOR|ORG)_[A-Za-z0-9]+', value))

def attached_text(post):
    parts = []
    flags = {}
    for field in ('article', 'entity', 'celebration', 'document', 'poll'):
        block = post.get(field)
        texts = []
        if isinstance(block, dict):
            for key in ('title', 'subtitle', 'text', 'transcript'):
                value = block.get(key)
                if isinstance(value, str) and value.strip():
                    texts.append(value.strip())
                elif isinstance(value, list):
                    texts.extend((item.strip() for item in value if isinstance(item, str) and item.strip()))
            if field == 'poll' and isinstance(block.get('options'), list):
                for option in block['options']:
                    if isinstance(option, str):
                        texts.append(option)
                    elif isinstance(option, dict):
                        for key in ('text', 'title'):
                            if isinstance(option.get(key), str):
                                texts.append(option[key])
        flags[f'attached_has_{field}'] = float(bool(texts))
        parts.extend(texts)
    document = post.get('document')
    flags['attached_has_transcript'] = float(isinstance(document, dict) and bool(document.get('transcript')))
    return ('\n\n'.join(dict.fromkeys(parts)), flags)

def load_context(data):
    rows = []
    discarded_person_references = 0
    for path in sorted((Path(data) / 'posts').glob('AUTHOR_*.json')):
        feed = json.loads(path.read_text())
        owner = feed['author']
        for post in feed['posts']:
            if not isinstance(post.get('author'), dict) or post['author'].get('code') != owner:
                continue
            shared = post.get('resharedPost')
            quoted = shared.get('text') if isinstance(shared, dict) else ''
            quoted = quoted if isinstance(quoted, str) else ''
            source_code = ''
            if isinstance(shared, dict):
                for field in ('company', 'author'):
                    block = shared.get(field)
                    code = block.get('code') if isinstance(block, dict) else None
                    if global_code(code):
                        source_code = code
                        break
                    discarded_person_references += int(isinstance(code, str) and code.startswith('PERSON_'))
            references = set()
            if source_code and source_code != owner:
                references.add(source_code)
            for field in ('companyMentions', 'mentions'):
                values = post.get(field)
                for value in values if isinstance(values, list) else []:
                    code = value.get('code') if isinstance(value, dict) else None
                    if global_code(code) and code != owner:
                        references.add(code)
                    discarded_person_references += int(isinstance(code, str) and code.startswith('PERSON_'))
            for code in re.findall('[⟨<]((?:AUTHOR|ORG)_[A-Za-z0-9]+)[⟩>]', post.get('text') or ''):
                if code != owner:
                    references.add(code)
            attached, flags = attached_text(post)
            if isinstance(shared, dict):
                extra, shared_flags = attached_text(shared)
                attached = '\n\n'.join(dict.fromkeys((s for s in (attached, extra) if s)))
                flags = {key: max(value, shared_flags[key]) for key, value in flags.items()}
            rows.append({'author': owner, 'post': post['urn'], 'quoted': quoted, 'attached': attached, 'references': sorted(references), 'quote_code': source_code, **flags})
    result = pd.DataFrame(rows)
    if result[['author', 'post']].duplicated().any():
        raise ValueError('Duplicate source context keys')
    return (result, discarded_person_references)

class ReferenceGraph:

    def fit(self, frame):
        self.references, self.incoming, self.post_count = ({}, Counter(), Counter())
        edges = Counter()
        for row in frame.itertuples():
            for target in row.references:
                self.references.setdefault(target, set()).add(row.author)
                self.incoming[target] += 1
                self.post_count[target] += 1
                edges[row.author, target] += 1
        keys = sorted({key for edge in edges for key in edge})
        self.rank, self.component, self.degree = ({}, {}, {})
        if keys:
            mapping = {key: i for i, key in enumerate(keys)}
            left, right, weight = ([], [], [])
            for (a, b), n in edges.items():
                left.append(mapping[a])
                right.append(mapping[b])
                weight.append(n)
            matrix = sparse.csr_matrix((np.asarray(weight, float), (left, right)), shape=(len(keys), len(keys)))
            outgoing = np.asarray(matrix.sum(1)).ravel()
            transition = sparse.diags(1 / np.maximum(outgoing, 1)) @ matrix
            rank = np.full(len(keys), 1 / len(keys))
            for _ in range(100):
                updated = 0.15 / len(keys) + 0.85 * (transition.T @ rank + rank[outgoing == 0].sum() / len(keys))
                if np.abs(updated - rank).sum() < 1e-09:
                    rank = updated
                    break
                rank = updated
            undirected = matrix + matrix.T
            _, labels = connected_components(undirected, directed=False)
            sizes = np.bincount(labels)
            degrees = np.diff(undirected.indptr)
            self.rank = dict(zip(keys, rank))
            self.component = {key: int(sizes[labels[i]]) for key, i in mapping.items()}
            self.degree = dict(zip(keys, degrees))
        return self

    def transform(self, frame):
        output = []
        for row in frame.itertuples():
            targets = row.references
            authors = Counter((author for target in targets for author in self.references.get(target, set())))
            counts = [len(self.references.get(target, set())) for target in targets]
            posts = [self.post_count.get(target, 0) for target in targets]
            ranks = [self.rank.get(target, 0.0) for target in targets]
            degrees = [self.degree.get(target, 0) for target in targets]
            output.append([len(targets), sum((n > 0 for n in counts)) / max(1, len(targets)), np.mean(counts) if counts else 0.0, max(counts, default=0), np.mean(posts) if posts else 0.0, max(posts, default=0), np.mean(ranks) if ranks else 0.0, max(ranks, default=0.0), max((self.component.get(target, 0) for target in targets), default=0), np.mean(degrees) if degrees else 0.0, max(degrees, default=0), self.incoming.get(row.author, 0), self.rank.get(row.author, 0.0), sum((n > 1 for n in authors.values())) / max(1, len(authors)), float(bool(row.quote_code)), float(any((t.startswith('ORG_') for t in targets))), float(any((t.startswith('AUTHOR_') for t in targets)))])
        return np.asarray(output, np.float32)
