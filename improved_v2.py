"""Lecture-inspired author context + early/late text fusion.

python improved_v2.py --stage prepare
python improved_v2.py --stage validate
python improved_v2.py --stage final

Optional frozen encoder arrays: pass --embeddings PATH to validate and final.
Default cache results/v2_cache is excluded from Git by the original .gitignore.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, confusion_matrix
from improved import build_table, tabular_model, text_model, NUMERIC, EXTRA, predict
from kairos.data import DATA_DIR, load_posts
from kairos.label import author_level
from kairos.metrics import score

SEEDS = (42, 123, 2026)

def build_context(data):
    table = build_table(data)
    p = author_level(load_posts(data)).reset_index(drop=True)
    if p.duplicated(['author', 'posted_at']).any():
        raise ValueError('Equal timestamps require a strict-past aggregation policy.')
    p['n_chars'] = p.text.str.len()
    p['n_words'] = p.text.str.split().str.len()
    g = p.groupby('author', sort=False)
    extra = []
    for col in ['n_chars', 'n_words']:
        past = g[col].transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
        name = f'{col}_relative5'
        p[name] = p[col] / (past + 1)
        extra.append(name)
    for n in [3, 10]:
        for stat in ['std', 'median', 'min', 'max']:
            name = f'past_{stat}{n}'
            p[name] = g.log_reactions.transform(
                lambda s: getattr(s.shift().rolling(n, min_periods=2), stat)())
            extra.append(name)
    p['previous_reaction_vs_level'] = g.log_reactions.shift() - p.level
    p['previous_resid'] = g.resid.shift()
    p['past_mean_resid5'] = g.resid.transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    p['format_frequency'] = p.groupby(['author', 'content_type']).cumcount() / p.hist_n.clip(lower=1)
    extra += ['previous_reaction_vs_level', 'previous_resid', 'past_mean_resid5', 'format_frequency']
    rows = []
    for f in sorted((Path(data) / 'posts').glob('AUTHOR_*.json')):
        feed = json.loads(f.read_text())
        a = feed['author']
        for post in feed['posts']:
            if not isinstance(post.get('author'), dict) or post['author'].get('code') != a:
                continue
            doc = post.get('document') or {}
            doc = doc if isinstance(doc, dict) else {}
            imgs = post.get('images') or []
            imgs = imgs if isinstance(imgs, list) else []
            rows.append(dict(author=a, post=post['urn'], image_count=len(imgs),
                document_pages=doc.get('totalPageCount', 0),
                has_document_transcript=int(bool(doc.get('transcript'))),
                has_reshared_post=int(bool(post.get('resharedPost'))),
                is_brand_partnership=int(bool(post.get('isBrandPartnership')))))
    meta = pd.DataFrame(rows)
    table = table.merge(p[['author', 'post'] + extra], on=['author', 'post'], validate='one_to_one')
    table = table.merge(meta, on=['author', 'post'], validate='one_to_one')
    return table, extra + list(meta.columns[2:])

def key_digest(df):
    keys = df[['author', 'post']].to_csv(index=False).encode()
    return hashlib.sha256(keys).hexdigest()

def tab(cols):
    model = tabular_model(True)
    prep = model.named_steps['prep']
    prep.transformers[0] = ('num', prep.transformers[0][1], cols)
    return model

def add_vectors(frame, values, prefix):
    names = [f'{prefix}_{i}' for i in range(values.shape[1])]
    result = pd.concat([frame.reset_index(drop=True), pd.DataFrame(values, columns=names)], axis=1)
    return result, names

def probabilities(tr, va, cols, embeddings=None, only=None):
    """Fit every learned transformation on tr, never on va or the official test."""
    probs = {}
    tm = text_model().fit(tr.text, tr.label)
    probs['tfidf'] = tm.predict_proba(va.text)
    if only is None or only == 'v1':
        probs['v1'] = tab(NUMERIC + EXTRA).fit(tr, tr.label).predict_proba(va)
    if only is None or only in ('context', 'semantic_late'):
        probs['context'] = tab(cols).fit(tr, tr.label).predict_proba(va)
    if only is None or only in ('lsa', 'combined'):
        svd = TruncatedSVD(n_components=64, n_iter=5, random_state=42)
        ztr = svd.fit_transform(tm.named_steps['tfidf'].transform(tr.text))
        zva = svd.transform(tm.named_steps['tfidf'].transform(va.text))
        norm = Normalizer()
        tt, names = add_vectors(tr, norm.fit_transform(ztr), 'lsa')
        vv, _ = add_vectors(va, norm.transform(zva), 'lsa')
        probs['lsa'] = tab(cols + names).fit(tt, tr.label).predict_proba(vv)
    if embeddings is not None:
        etr, eva = embeddings[tr.row_id], embeddings[va.row_id]
        if only is None or only == 'semantic_late':
            lm = LogisticRegression(C=1, class_weight='balanced', max_iter=500).fit(etr, tr.label)
            probs['semantic_linear'] = lm.predict_proba(eva)
        if only is None or only in ('semantic', 'combined'):
            pca = PCA(n_components=64, random_state=42)
            tt, names = add_vectors(tr, pca.fit_transform(etr), 'semantic')
            vv, _ = add_vectors(va, pca.transform(eva), 'semantic')
            probs['semantic'] = tab(cols + names).fit(tt, tr.label).predict_proba(vv)
    if 'lsa' in probs and 'semantic' in probs:
        probs['combined'] = .5 * probs['lsa'] + .5 * probs['semantic']
    return probs

def candidates(probs, y):
    rows = []
    for name in ['context', 'lsa'] + (['semantic', 'semantic_late', 'combined'] if 'semantic' in probs else []):
        alphas = [.15, .3, .5] if name == 'semantic_late' else [0., .15, .3]
        a, b = (probs['context'], probs['semantic_linear']) if name == 'semantic_late' else (probs[name], probs['tfidf'])
        for alpha in alphas:
            for weight in [1., 1.2, 1.4, 1.6]:
                pred = predict(a, b, alpha, weight)
                rows.append(dict(model=name, text_weight=alpha, typical_weight=weight,
                    macro_f1=float(f1_score(y, pred, average='macro'))))
    return rows

def select(reports):
    all_rows = pd.DataFrame([dict(seed=r['seed'], **c) for r in reports for c in r['candidates']])
    means = all_rows.groupby(['model', 'text_weight', 'typical_weight'], sort=False).macro_f1.mean()
    model, alpha, weight = means.idxmax()
    return dict(model=model, text_weight=float(alpha), typical_weight=float(weight),
                mean_validation_macro_f1=float(means.max()))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', choices=['prepare', 'validate', 'final', 'all'], default='all')
    ap.add_argument('--data', default=DATA_DIR)
    ap.add_argument('--cache', type=Path, default=Path('results/v2_cache'))
    ap.add_argument('--output', type=Path, default=Path('lab1_submission_v2'))
    ap.add_argument('--embeddings', type=Path)
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage in ('prepare', 'all') or not (args.cache/'table.pkl').exists():
        df, context = build_context(args.data)
        df.to_pickle(args.cache/'table.pkl')
        (args.cache/'context_columns.json').write_text(json.dumps(context))
        with (args.cache/'texts.jsonl').open('w') as f:
            for row in df[['author', 'post', 'text']].to_dict('records'):
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
        if args.stage == 'prepare':
            print('Prepared data in', args.cache)
            return
    df = pd.read_pickle(args.cache/'table.pkl')
    df['row_id'] = np.arange(len(df))
    context = json.loads((args.cache/'context_columns.json').read_text())
    cols = NUMERIC + EXTRA + context
    train, test = df[df.split == 'train'], df[df.split == 'test']
    assert not set(train.author) & set(test.author)
    emb = None
    if args.embeddings:
        emb = np.load(args.embeddings, mmap_mode='r')
        cfg = json.loads(args.embeddings.with_name('embedding_config.json').read_text())
        if cfg['key_sha256'] != key_digest(df) or emb.shape != (len(df), 384) or not np.isfinite(emb).all():
            raise ValueError('Embeddings are missing, invalid or in a different row order')
    if args.stage in ('validate', 'all'):
        reports = []
        for seed in SEEDS:
            ti, vi = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(train, groups=train.author))
            tr, va = train.iloc[ti], train.iloc[vi]
            assert not set(tr.author) & set(va.author)
            probs = probabilities(tr, va, cols, emb)
            incumbent = score(va.label, predict(probs['v1'], probs['tfidf'], .3, 1.4))
            report = dict(seed=seed, incumbent=incumbent, candidates=candidates(probs, va.label),
                          validation_authors=sorted(va.author.unique().tolist()))
            reports.append(report)
            print('Validation', seed, 'incumbent', incumbent,
                  'best', max(report['candidates'], key=lambda c: c['macro_f1']), flush=True)
        selection = dict(selected=select(reports), folds=reports, key_sha256=key_digest(df),
            used_embeddings=bool(args.embeddings), python=platform.python_version(), sklearn=sklearn.__version__)
        (args.output/'selection.json').write_text(json.dumps(selection, indent=2))
        print('Frozen selection:', selection['selected'], flush=True)
        if args.stage == 'validate':
            return
    selection = json.loads((args.output/'selection.json').read_text())
    if selection['key_sha256'] != key_digest(df):
        raise ValueError('Data changed since validation')
    best = selection['selected']
    if (best['model'].startswith('semantic') or best['model']=='combined') and emb is None:
        raise ValueError('Selected model requires --embeddings')
    probs = probabilities(train, test, cols, emb, only=best['model'])
    a, b = (probs['context'], probs['semantic_linear']) if best['model']=='semantic_late' else (probs[best['model']], probs['tfidf'])
    pred = predict(a, b, best['text_weight'], best['typical_weight'])
    test[['author', 'post']].assign(label=pred).to_csv(args.output/'predictions.csv', index=False)
    metrics = score(test.label, pred)
    metrics['f1_by_class'] = f1_score(test.label, pred, labels=[0,1,2], average=None).tolist()
    metrics['confusion_matrix'] = confusion_matrix(test.label, pred, labels=[0,1,2]).tolist()
    result = dict(selection=best, test=metrics, n_test=len(test), python=platform.python_version(), sklearn=sklearn.__version__)
    (args.output/'metrics.json').write_text(json.dumps(result, indent=2))
    print('Final test:', metrics, flush=True)

if __name__ == '__main__':
    main()
