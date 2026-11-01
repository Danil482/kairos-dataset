from collections import Counter
from itertools import combinations
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from kairos import features
from kairos.data import load_posts, load_followers
from kairos.label import author_level
from kairos.text_features import prepare_texts, _safe_surface, _nfkc
from kairos.history_features import build_text_history
from kairos.engagement_history import engagement_history
from kairos.extended_features import semantic_reaction_history
from kairos.contextual_features import semantic_history
from kairos.graph_features import hashtags, TextGraphReference
from kairos.quote_features import load_context, ReferenceGraph
from kairos.causal_history import build_causal_history
from kairos.encoder import encode


ROOT = Path(__file__).resolve().parent


def fit_tags(texts, authors):
    model = TextGraphReference()
    model.n_docs = len(texts)
    model.tag_df, model.pair_df = Counter(), Counter()
    model.tag_authors = {}
    for text, author in zip(texts, authors):
        tags = sorted(set(hashtags(text)))[:32]
        model.tag_df.update(tags)
        model.pair_df.update(combinations(tags, 2))
        for tag in tags:
            model.tag_authors.setdefault(tag, set()).add(author)
    model._centralities()
    return model



def transform_tags(model, texts):
    base = model._tag_features(texts).to_numpy()
    rows = []
    for text in texts:
        tags = sorted(set(hashtags(text)))[:32]
        groups = [model.tag_authors.get(tag, set()) for tag in tags]
        counts = np.asarray([len(group) for group in groups], float)
        authors = Counter(author for group in groups for author in group)
        total = sum(authors.values())
        concentration = sum((n/max(1, total))**2 for n in authors.values())
        rows.append([counts.mean() if len(counts) else 0., counts.max() if len(counts) else 0., len(authors), concentration,
            sum(n>1 for n in authors.values())/max(1, len(authors)), float((counts>0).mean()) if len(counts) else 0.])
    return np.hstack([base, np.asarray(rows, np.float32)])


def matrix(groups, schema, names, extra=None):
    arrays = []
    lookup = {}
    offset = 0
    for group, values in groups.items():
        values = np.nan_to_num(np.asarray(values, np.float32), nan=0., posinf=0., neginf=0.)
        arrays.append(values)
        lookup.update({f'{group}__{name}': offset+i for i, name in enumerate(schema[group])})
        offset += values.shape[1]
    if extra is not None:
        arrays.append(extra.to_numpy(dtype=np.float32))
        lookup.update({f'causal__{name}': offset+i for i, name in enumerate(extra.columns)})
    return np.ascontiguousarray(np.hstack(arrays)[:, [lookup[name] for name in names]])


def prepare(data, weights, device, split='test'):
    recipe = json.loads((weights/'recipe.json').read_text())
    labels = pd.read_csv(data/'labels.csv', usecols=['author', 'post', 'split'])
    ids = pd.read_csv(data/'train_ids.csv') if split == 'train' else labels.loc[labels.split.eq('test'), ['author', 'post']].reset_index(drop=True)
    keys = pd.MultiIndex.from_frame(ids)
    posts = load_posts(data)
    indexed = posts.set_index(['author', 'post'])
    current = indexed.reindex(keys).reset_index()
    if current.text.isna().any() or keys.has_duplicates:
        raise ValueError('Missing or duplicate test posts')
    authors = set(ids.author)
    pool = posts.loc[posts.author.isin(authors)].reset_index(drop=True)
    pool_keys = pd.MultiIndex.from_frame(pool[['author', 'post']])
    current_rows = pool_keys.get_indexer(keys)
    surface, _, style = prepare_texts(current.text)
    baseline = features.build(author_level(posts), load_followers(data)).set_index(['author', 'post'])
    reactions = engagement_history(pool).set_index(['author', 'post'])
    snapshot = baseline.join(reactions).reindex(keys)
    chronological = pool[['author', 'post']].join(build_text_history(pool)).set_index(['author', 'post']).reindex(keys)
    all_context, _ = load_context(data)
    context = all_context.set_index(['author', 'post']).reindex(keys).reset_index()
    _, _, quoted_style = prepare_texts(context.quoted)
    quoted_style['quote_has_text'] = context.quoted.str.strip().ne('').astype(float)
    _, _, attached_style = prepare_texts(context.attached)
    for name in ('article', 'entity', 'celebration', 'document', 'poll', 'transcript'):
        attached_style[f'attached_has_{name}'] = context[f'attached_has_{name}']
    cutoffs = json.loads((data/'split.json').read_text())['cutoffs_on_log_residual']
    extra = build_causal_history(pool, cutoffs).reindex(keys)
    if extra.isna().any().any():
        raise ValueError('Missing causal history')
    quoted = context[['author', 'post', 'quoted']].rename(columns={'quoted': 'text'}).copy()
    quoted['post'] = quoted.post+'::quoted'
    frames = pd.concat([pool[['author', 'post', 'text']], quoted], ignore_index=True)
    matrices = {}
    for kind in ('base', 'large'):
        print('Preparing', kind, flush=True)
        schema = recipe['schemas'][kind]
        transform = weights/'transforms'/kind
        vectors = encode(frames, recipe['encoders'][kind], data/'embeddings'/kind, device)
        pool_vectors = vectors[:len(pool)]
        root = pool_vectors[current_rows]
        quote_vectors = vectors[len(pool):]
        pca = joblib.load(transform/'context_pca.joblib')
        projected = normalize(pca.transform(pool_vectors).astype(np.float32))
        projected[np.linalg.norm(pool_vectors, axis=1)<1e-8] = 0
        history, _ = semantic_history(pool, projected, pool.posted_at)
        numerical = [name for name in schema['content'] if not name.startswith('content_type_')]
        categories = [name[len('content_type_'):] for name in schema['content'] if name.startswith('content_type_')]
        content = np.hstack([style[numerical].to_numpy(), np.column_stack([(current.content_type==name).to_numpy(float) for name in categories])])
        semantic = semantic_reaction_history(ids, pool_vectors, posts, pool[['author', 'post']])
        if split == 'train':
            authors = np.asarray(sorted(ids.author.unique()))
            np.random.default_rng(42).shuffle(authors)
            folds = {author: fold for fold, group in enumerate(np.array_split(authors, 3)) for author in group}
            tags = np.zeros((len(ids), len(schema['tags'])), np.float32)
            for fold in range(3):
                held = ids.author.map(folds).eq(fold).to_numpy()
                reference = fit_tags(surface.loc[~held], ids.author.loc[~held])
                tags[held] = transform_tags(reference, surface.loc[held])
        else:
            tags = transform_tags(joblib.load(transform/'tag_reference.joblib'), surface)
        quote_pca = joblib.load(transform/'quote_pca.joblib')
        quote_projected = normalize(quote_pca.transform(quote_vectors).astype(np.float32))
        quote_projected[np.linalg.norm(quote_vectors, axis=1)<1e-8] = 0
        similarity = np.sum(root*quote_vectors, axis=1, keepdims=True)
        attached = joblib.load(transform/'attached_transformer.joblib')
        strings = context.attached.map(lambda text: _safe_surface(_nfkc(text))[0])
        attached_projected = normalize(attached['svd'].transform(attached['vectorizer'].transform(strings)).astype(np.float32))
        if split == 'train':
            graph = np.zeros((len(ids), len(schema['reference_graph'])), np.float32)
            for fold in range(3):
                held = ids.author.map(folds).eq(fold).to_numpy()
                graph[held] = ReferenceGraph().fit(context.loc[~held]).transform(context.loc[held])
        else:
            graph = joblib.load(transform/'reference_graph.joblib').transform(context)
        groups = {'content': content, 'chronology': chronological[schema['chronology']].to_numpy(),
            'snapshot': snapshot[schema['snapshot']].to_numpy(), 'embedding': root, 'context': history[current_rows], 'tags': tags,
            'semantic_reaction': semantic.to_numpy(), 'quote_pca': np.hstack([quote_projected, quoted_style.to_numpy(), similarity]),
            'attached': np.hstack([attached_projected, attached_style.to_numpy()]), 'reference_graph': graph}
        matrices[kind] = matrix(groups, schema, recipe['feature_names'][kind])
        if kind == 'large':
            matrices['new'] = matrix(groups, schema, recipe['feature_names']['new'], extra)
    return ids, matrices, surface
