"""Cross-check and reproduce class decision multipliers for frozen V2 models.

The original model probabilities are never modified. This is a decision rule,
not a claim that probabilities have become calibrated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kairos.experiment import ROOT, metrics, save_json
from kairos.experiment_selection import passes_gate
from kairos.experiment_uncertainty import paired_author_bootstrap

GRID = tuple(np.round(np.arange(0.8, 1.201, 0.05), 2).tolist())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decide(probabilities, weights):
    p, w = np.asarray(probabilities), np.asarray(weights, dtype=float)
    if (p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all()
            or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1)):
        raise ValueError("Expected finite three-class probabilities summing to one")
    if w.shape != (3,) or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("Expected three positive finite decision multipliers")
    return (p * w).argmax(1)


def macro(y, pred):
    confusion = np.bincount(np.asarray(y, dtype=int) * 3 + pred, minlength=9).reshape(3, 3)
    denominator = confusion.sum(0) + confusion.sum(1)
    return float(np.divide(2 * np.diag(confusion), denominator,
                          out=np.zeros(3, dtype=float), where=denominator > 0).mean())


def best_weights(y, probabilities, mask):
    truth = np.asarray(y, dtype=int)[mask]
    p = np.asarray(probabilities)[mask]
    if not len(truth) or not np.isin(truth, [0, 1, 2]).all():
        raise ValueError("A nonempty calibration subset with valid labels is required")
    candidates = [(macro(truth, decide(p, [a, 1, b])), -abs(a-1)-abs(b-1), a, b)
                  for a in GRID for b in GRID]
    score, _, a, b = max(candidates)
    return np.asarray([a, 1.0, b]), score


def cross_calibrate(y, probabilities, fold, authors):
    y, fold, authors = np.asarray(y, dtype=int), np.asarray(fold), np.asarray(authors)
    p = np.asarray(probabilities)
    if not (len(y) == len(p) == len(fold) == len(authors)):
        raise ValueError("OOF arrays must align")
    decide(p, [1, 1, 1])
    if set(np.unique(fold)) != {0, 1, 2} or not np.isin(y, [0, 1, 2]).all():
        raise ValueError("Expected the three original folds and labels 0/1/2")
    if pd.DataFrame({'author': authors, 'fold': fold}).groupby('author').fold.nunique().max() != 1:
        raise ValueError("An author crosses outer folds")
    predicted = np.empty(len(y), dtype=int)
    halves = []
    for f in np.unique(fold):
        ids = np.unique(authors[fold == f])
        if len(ids) < 2:
            raise ValueError("Each outer fold needs at least two authors")
        ids = ids[np.random.default_rng(42 + int(f)).permutation(len(ids))]
        side = np.isin(authors, ids[::2])
        for h in (0, 1):
            evaluation = (fold == f) & (side == bool(h))
            calibration = (fold == f) & (side != bool(h))
            weights, fitted_score = best_weights(y, p, calibration)
            predicted[evaluation] = decide(p[evaluation], weights)
            halves.append({'fold': int(f), 'half': h, 'weights': weights.tolist(),
                           'calibration_authors': sorted(set(authors[calibration])),
                           'evaluation_authors': sorted(set(authors[evaluation])),
                           'calibration_macro_f1': fitted_score,
                           'evaluation_metrics': metrics(y[evaluation], predicted[evaluation]),
                           'reference_metrics': metrics(y[evaluation], p[evaluation].argmax(1))})
    return predicted, halves


def search(base, output, config_path):
    with np.load(base / 'oof_probabilities.npz', allow_pickle=False) as z:
        y, p, fold, authors = z['y'], z['selected'], z['fold'], z['authors']
    previous = json.loads((base / 'selection.json').read_text())
    pinned = json.loads((ROOT / 'configs/stage2_folds.json').read_text())['fold_authors']
    for f in range(3):
        if set(authors[fold == f]) != set(pinned[str(f)]):
            raise ValueError("OOF author assignments do not match the pinned folds")
    if not np.isclose(macro(y, p.argmax(1)), previous['metrics']['macro_f1'], atol=1e-12, rtol=0):
        raise ValueError("OOF predictions do not match the selected base model")
    predicted, halves = cross_calibrate(y, p, fold, authors)
    gate = passes_gate(y, np.eye(3)[predicted], p, fold)
    weights, fitted_score = best_weights(y, p, np.ones(len(y), dtype=bool))
    if not gate['accepted']:
        weights = np.ones(3)
        fitted_score = macro(y, p.argmax(1))
    report = {'base_run_id': previous['run_id'], 'grid': list(GRID), 'grid_pairs': len(GRID)**2,
              'seed': 42, 'reference_metrics': metrics(y, p.argmax(1)),
              'cross_calibration_metrics': metrics(y, predicted), 'gate': gate,
              'halves': halves, 'weights': weights.tolist(),
              'full_oof_refit_macro_f1': fitted_score,
              'full_oof_refit_is_validation': False,
              'paired_author_bootstrap': paired_author_bootstrap(y, predicted, p.argmax(1), authors),
              'test_labels_used': False, 'oof_sha256': sha(base/'oof_probabilities.npz'),
              'source_sha256': sha(__file__),
              'interpretation': 'Cross-calibration of an already selected ensemble. Each evaluation author is excluded from coefficient fitting and base fitting; calibration uses only the other half of the same outer validation fold. Prior ensemble-selection bias remains. Six fitted rules validate the procedure; final fixed weights are refitted on all OOF rows.',
              'grid_boundary_note': 'Selected lower-bound weights are retained; no post-hoc grid expansion.'}
    output.mkdir(parents=True, exist_ok=True)
    save_json(output/'search.json', report)
    np.savez_compressed(output/'cross_calibration.npz', predicted=predicted, y=y, fold=fold, authors=authors)
    config = {'base_config': 'configs/solution_v2.json',
              'base_config_sha256': sha(ROOT/'configs/solution_v2.json'),
              'base_output': 'results/v2', 'output': 'results/v2_decision',
              'weights': weights.tolist(), 'search_report_sha256': sha(output/'search.json')}
    save_json(config_path, config)
    print(json.dumps({k: report[k] for k in ('weights', 'gate', 'cross_calibration_metrics', 'full_oof_refit_macro_f1')}, indent=2))


def apply(config, base, output):
    original = base/'final'
    manifest = json.loads((original/'manifest.json').read_text())
    if sha(original/'predictions.csv') != manifest['predictions_sha256']:
        raise ValueError("Original prediction checksum mismatch")
    frame = pd.read_csv(original/'predictions.csv')
    p = np.load(original/'probabilities.npy', allow_pickle=False)
    if len(frame) != len(p) or frame.duplicated(['author', 'post']).any():
        raise ValueError("Prediction keys must be unique and aligned")
    if not np.array_equal(frame.label.to_numpy(), decide(p, [1, 1, 1])):
        raise ValueError("Original predictions and probabilities do not align")
    base_config = json.loads((ROOT/config['base_config']).read_text())
    if manifest['selected'] != base_config['selected']:
        raise ValueError("Base model recipe differs from the frozen configuration")
    predicted = decide(p, config['weights'])
    frame['label'] = predicted
    target = output/'final'; target.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target/'predictions.csv', index=False)
    np.save(target/'decision_scores.npy', p*np.asarray(config['weights']), allow_pickle=False)
    save_json(target/'manifest.json', {'weights': config['weights'], 'n_rows': len(frame),
              'base_run_id': manifest['run_id'], 'base_selected': manifest['selected'],
              'base_predictions_sha256': manifest['predictions_sha256'],
              'base_probabilities_sha256': sha(original/'probabilities.npy'),
              'predictions_sha256': sha(target/'predictions.csv'),
              'search_report_sha256': config['search_report_sha256'],
              'source_sha256': sha(__file__), 'test_labels_used': False,
              'decision_scores_are_probabilities': False})
    print(f"Saved {len(frame)} predictions with decision weights {config['weights']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=['search', 'apply', 'reproduce'], default='reproduce')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/solution_v2_decision.json')
    parser.add_argument('--base-output', type=Path, default=ROOT/'results/v2')
    parser.add_argument('--output', type=Path, default=ROOT/'results/v2_decision')
    parser.add_argument('--data', type=Path, default=ROOT/'data')
    args = parser.parse_args()
    if args.action == 'search':
        search(args.base_output, args.output, args.config)
        return
    config = json.loads(args.config.read_text())
    if sha(ROOT/config['base_config']) != config['base_config_sha256']:
        raise ValueError("The frozen base configuration changed")
    base = args.base_output
    if args.action == 'reproduce':
        base = args.output/'base'
        subprocess.run([sys.executable, '-m', 'kairos.experiment', '--config',
                        str(ROOT/config['base_config']), '--data', str(args.data),
                        '--output', str(base)], cwd=ROOT, check=True)
    apply(config, base, args.output)


if __name__ == '__main__':
    main()
