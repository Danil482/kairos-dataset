from pathlib import Path
import argparse
import json
import sys
sys.modules.setdefault('IPython', None)
import joblib
import numpy as np
from scipy import sparse
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from prepare import prepare


ROOT = Path(__file__).resolve().parent


def predict(data, weights, output, device):
    recipe = json.loads((weights/'recipe.json').read_text())
    ids, matrices, texts = prepare(data, weights, device)
    probabilities = {}
    for key in recipe['weights']:
        model = joblib.load(weights/f'models/{key}.joblib')
        if key == 'lexical':
            values = sparse.hstack([model['word'].transform(texts), model['char'].transform(texts)], format='csr')
            probability = softmax(model['model'].decision_function(values)*2, axis=1)
        else:
            probability = model.predict_proba(matrices[key if key in ('base', 'large') else 'new'])
        probabilities[key] = np.asarray(probability, np.float32)
        print('Predicted', key, flush=True)
    scores = sum(probabilities[key]*weight for key, weight in recipe['weights'].items())*np.asarray(recipe['multipliers'])
    if not np.isfinite(scores).all() or scores.shape != (len(ids), 3):
        raise ValueError('Invalid model scores')
    output.parent.mkdir(parents=True, exist_ok=True)
    ids.assign(label=scores.argmax(1)).to_csv(output, index=False)
    print(output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT/'data')
    parser.add_argument('--weights', type=Path, default=ROOT/'weights')
    parser.add_argument('--output', type=Path, default=ROOT/'submission/submission.csv')
    parser.add_argument('--device', choices=['auto', 'mps', 'cpu'], default='auto')
    args = parser.parse_args()
    with threadpool_limits(limits=3):
        predict(args.data, args.weights, args.output, args.device)
