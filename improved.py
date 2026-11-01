"""Author-disjoint validation, text/history ablation, then one final test.
Run from the repository: python improved.py
No new dependencies beyond requirements.txt. Never fits on test labels.
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from baseline import model as baseline_model, NUMERIC
from kairos import features
from kairos.data import DATA_DIR, load_posts, load_followers, load_labels
from kairos.label import author_level
from kairos.metrics import score

SEED = 42
EXTRA = ['recent3', 'recent5', 'recent10', 'recent_std5', 'trend5',
         'word_length', 'exclamations', 'upper_ratio']

def clean_text(text):
    # Do not learn a vocabulary of pseudonymous identities.
    return re.sub(r'⟨[^⟩]+⟩', ' placeholder ', text).lower()

def text_model():
    return Pipeline([
        ('tfidf', TfidfVectorizer(preprocessor=clean_text, lowercase=True,
          ngram_range=(1, 2), min_df=5, max_df=.98, max_features=40000,
          sublinear_tf=True, strip_accents='unicode')),
        ('clf', LogisticRegression(C=1., class_weight='balanced',
                                  solver='lbfgs', max_iter=500))])

def tabular_model(enriched=False):
    m = baseline_model()
    if enriched:
        transformers = m.named_steps['prep'].transformers
        transformers[0] = ('num', transformers[0][1], NUMERIC + EXTRA)
    return m

def build_table(data):
    posts = author_level(load_posts(data))
    df = features.build(posts, load_followers(data))
    g = df.groupby('author')['log_reactions']
    for n in (3, 5, 10):
        df[f'recent{n}'] = g.transform(lambda x: x.shift().rolling(n, min_periods=1).mean())
    df['recent_std5'] = g.transform(lambda x: x.shift().rolling(5, min_periods=2).std())
    df['trend5'] = df['recent5'] - df['level']
    df['word_length'] = df.n_chars / df.n_words.clip(lower=1)
    df['exclamations'] = df.text.str.count('!')
    df['upper_ratio'] = df.text.str.count('[A-Z]') / df.n_chars.clip(lower=1)
    # Explicit feature whitelist excludes current reactions, residual and labels.
    keep = ['author', 'post', 'text'] + NUMERIC + features.CONTENT_CAT + EXTRA
    table = load_labels(data).merge(df[keep], on=['author', 'post'], validate='one_to_one')
    if len(table) != len(load_labels(data)):
        raise ValueError('Some labelled posts are missing')
    return table

def predict(p_tab, p_text, alpha, typical_weight):
    p = ((1-alpha)*p_tab + alpha*p_text).copy()
    p[:, 1] *= typical_weight
    return p.argmax(axis=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=DATA_DIR)
    ap.add_argument('--output', default='lab1_submission')
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    table = build_table(args.data)
    train = table[table.split == 'train'].reset_index(drop=True)
    test = table[table.split == 'test'].reset_index(drop=True)
    ti, vi = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(train, groups=train.author))
    tr, va = train.iloc[ti], train.iloc[vi]
    assert not (set(tr.author) & set(va.author))
    assert not (set(train.author) & set(test.author))
    print(f'Validation: fit {len(tr)} posts / {tr.author.nunique()} authors; '
          f'validate {len(va)} posts / {va.author.nunique()} authors', flush=True)
    probabilities = {}; validation = {}
    for name, enriched in [('baseline', False), ('history_style', True)]:
        m = tabular_model(enriched).fit(tr, tr.label)
        assert list(m.classes_) == [0, 1, 2]
        probabilities[name] = m.predict_proba(va)
        validation[name] = score(va.label, probabilities[name].argmax(axis=1))
        print(name, validation[name], flush=True)
    tm = text_model().fit(tr.text, tr.label)
    tp = tm.predict_proba(va.text)
    validation['text_only'] = score(va.label, tp.argmax(axis=1))
    print('text_only', validation['text_only'], flush=True)
    candidates = []
    for name, p in probabilities.items():
        for alpha in (0., .15, .3, .5):
            for weight in (1., 1.2, 1.4, 1.6):
                pred = predict(p, tp, alpha, weight)
                candidates.append(dict(tabular=name, text_weight=alpha, typical_weight=weight,
                    macro_f1=float(f1_score(va.label, pred, average='macro'))))
    best = max(candidates, key=lambda x: x['macro_f1'])
    # Freeze selection and save evidence BEFORE accessing final test performance.
    report = dict(seed=SEED, validation_authors=sorted(va.author.unique().tolist()),
                  validation=validation, candidates=candidates, selected=best,
                  n_train=len(train), n_test=len(test))
    (out/'selection.json').write_text(json.dumps(report, indent=2))
    print('Selected on validation:', best, flush=True)
    final_tab = tabular_model(best['tabular']=='history_style').fit(train, train.label)
    p = final_tab.predict_proba(test)
    t = np.zeros_like(p)
    if best['text_weight']:
        final_text = text_model().fit(train.text, train.label)
        t = final_text.predict_proba(test.text)
    pred = predict(p, t, best['text_weight'], best['typical_weight'])
    test[['author','post']].assign(label=pred).to_csv(out/'predictions.csv', index=False)
    report['test'] = score(test.label, pred)
    report['test']['confusion_matrix'] = confusion_matrix(test.label, pred, labels=[0,1,2]).tolist()
    report['test']['f1_by_class'] = f1_score(test.label, pred, labels=[0,1,2], average=None).tolist()
    (out/'metrics.json').write_text(json.dumps(report, indent=2))
    print('Final test:', report['test'], flush=True)
    print('Saved to', out, flush=True)

if __name__ == '__main__':
    main()
