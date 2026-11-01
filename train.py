from pathlib import Path
import argparse
import hashlib
import json
import sys
sys.modules.setdefault('IPython', None)
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from prepare import prepare


ROOT = Path(__file__).resolve().parent


def train(data, weights, output, device):
    recipe = json.loads((weights/'recipe.json').read_text())
    if hashlib.sha256((data/'train_ids.csv').read_bytes()).hexdigest() != recipe['train_ids_sha256']:
        raise ValueError('Training ID order differs')
    saved = np.load(data/'train_weights.npz')
    ids = pd.read_csv(data/'train_ids.csv')
    labels = pd.read_csv(data/'labels.csv')
    labels = ids.merge(labels.loc[labels.split.eq('train'), ['author', 'post', 'label']], on=['author', 'post'], validate='one_to_one')
    y = labels.label.to_numpy()
    if not np.array_equal(y, saved['targets']):
        raise ValueError('Training targets differ')
    ids, matrices, texts = prepare(data, weights, device, split='train')
    output.mkdir(parents=True, exist_ok=True)
    for key in recipe['weights']:
        if key == 'lexical':
            word = TfidfVectorizer(ngram_range=(1, 2), max_features=65000, min_df=3, sublinear_tf=True, dtype=np.float32)
            char = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=70000, min_df=4, sublinear_tf=True, dtype=np.float32)
            matrix = sparse.hstack([word.fit_transform(texts), char.fit_transform(texts)], format='csr')
            counts = np.bincount(y, minlength=3)
            sample_weight = (len(y)/(3*np.maximum(counts, 1)))**.6
            sample_weight = sample_weight[y]
            sample_weight /= sample_weight.mean()
            estimator = RidgeClassifier(alpha=10., solver='lsqr', tol=.003)
            estimator.fit(matrix, y, sample_weight=sample_weight)
            model = {'word': word, 'char': char, 'model': estimator}
        else:
            config = recipe['configs'][key]
            matrix = matrices[key if key in ('base', 'large') else 'new']
            if config['kind'] == 'catboost':
                model = CatBoostClassifier(iterations=config['iterations'], depth=config['depth'], learning_rate=config['learning_rate'],
                    l2_leaf_reg=config['l2_leaf_reg'], loss_function='MultiClass', random_seed=42, thread_count=3, allow_writing_files=False)
            else:
                model = LGBMClassifier(**config['params'], n_estimators=config['iterations'], class_weight=None, metric='None',
                    objective='multiclass', random_state=42, n_jobs=3, verbosity=-1, deterministic=True, force_col_wise=True)
            model.fit(matrix, y, sample_weight=saved[key])
        joblib.dump(model, output/f'{key}.joblib', compress=3)
        print('Trained', key, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT/'data')
    parser.add_argument('--weights', type=Path, default=ROOT/'weights')
    parser.add_argument('--output', type=Path, default=ROOT/'weights/models')
    parser.add_argument('--device', choices=['auto', 'mps', 'cpu'], default='auto')
    args = parser.parse_args()
    with threadpool_limits(limits=3):
        train(args.data, args.weights, args.output, args.device)
