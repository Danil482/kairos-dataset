#!/usr/bin/env python3
"""Independent train-author OOF MLP experiment on frozen vectors and past-only features."""
import argparse
import importlib.util
import json
import math
import os
import shutil
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn

import kairos_lab.train as base
from kairos_lab.common import (artifacts_fingerprint, atomic_json, digest, identity,
                               mark, now, stop_check, verified_marker)
from kairos_lab.dataset import KEYS


class MLP(nn.Module):
    def __init__(self, width, hidden, dropout):
        super().__init__()
        layers = []
        for size in hidden:
            layers += [nn.Linear(width, size), nn.LayerNorm(size), nn.GELU(), nn.Dropout(dropout)]
            width = size
        self.layers = nn.Sequential(*layers, nn.Linear(width, 3))

    def forward(self, value):
        return self.layers(value)


def normalizer(values):
    """Fit on the job's training rows only; validation/test rows are never passed here."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        median = np.nanmedian(values, axis=0)
    median = np.nan_to_num(median, nan=0.0)
    filled = np.where(np.isfinite(values), values, median)
    mean, scale = filled.mean(0), filled.std(0)
    scale = np.maximum(scale, 1e-5)
    return {'median': median, 'mean': mean, 'scale': scale}


def normalize(values, state):
    filled = np.where(np.isfinite(values), values, state['median'])
    return np.clip((filled-state['mean'])/state['scale'], -10, 10).astype(np.float32)


def atomic_torch(path, value):
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def fit_network(values, labels, spec, folder, work, fingerprint, device, seed, deadline, until_epoch=None):
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = MLP(values.shape[1], spec['hidden'], spec['dropout']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
    latest = folder/'latest.json'
    start_epoch, elapsed, average, averaged = 0, 0.0, {}, 0
    preprocess = normalizer(values)
    if latest.exists():
        assert verified_marker(latest, fingerprint)
        marker = json.loads(latest.read_text())
        state = torch.load(folder/marker['checkpoint'], map_location='cpu', weights_only=False)
        assert state['fingerprint'] == fingerprint
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch, elapsed = state['epoch'], state['elapsed_seconds']
        preprocess, average, averaged = state['preprocess'], state['average'], state['averaged']
        torch.set_rng_state(state['torch_rng'])
        if device == 'cuda':
            torch.cuda.set_rng_state(state['cuda_rng'], device=0)
        rng.bit_generator.state = state['shuffle_rng']
    x = normalize(values, preprocess)
    y = np.asarray(labels, dtype=np.int64)
    weights = torch.tensor(base.class_weights(y, spec['class_power']), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    batches = math.ceil(len(x)/spec['batch_size'])
    end_epoch = min(spec['epochs'], until_epoch) if until_epoch is not None else spec['epochs']
    started = time.monotonic()
    previous_elapsed = elapsed
    for epoch in range(start_epoch, end_epoch):
        model.train()
        order = rng.permutation(len(x))
        for batch, offset in enumerate(range(0, len(order), spec['batch_size'])):
            stop_check(work)
            if time.time() >= deadline:
                raise base.BudgetReached('Hard deadline reached; last committed epoch retained')
            selected = order[offset:offset+spec['batch_size']]
            inputs = torch.from_numpy(x[selected]).to(device)
            targets = torch.from_numpy(y[selected]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite neural loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if batch % 20 == 0 or offset+spec['batch_size'] >= len(order):
                atomic_json(work/'neural-progress.json', {
                    'candidate': spec['name'], 'job': folder.name, 'epoch': epoch+1,
                    'iterations': epoch*batches+batch+1, 'total_iterations': spec['epochs']*batches,
                    'elapsed_seconds': previous_elapsed+time.monotonic()-started,
                    'session_iterations': (epoch-start_epoch)*batches+batch+1,
                    'session_elapsed_seconds': time.monotonic()-started, 'at': now()})
        if epoch >= spec['epochs']-spec['swa_last_epochs']:
            current = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            average = current if not averaged else {k: (average[k]*averaged+v)/(averaged+1) for k, v in current.items()}
            averaged += 1
        elapsed = previous_elapsed+time.monotonic()-started
        state = {'fingerprint': fingerprint, 'epoch': epoch+1, 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(), 'preprocess': preprocess,
                 'average': average, 'averaged': averaged, 'elapsed_seconds': elapsed,
                 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state(device=0) if device == 'cuda' else None,
                 'shuffle_rng': rng.bit_generator.state}
        checkpoint = folder/f'optimizer-epoch-{epoch+1:03d}.pt'
        atomic_torch(checkpoint, state)
        mark(latest, fingerprint, [checkpoint], checkpoint=checkpoint.name, epoch=epoch+1)
    if averaged:
        model.load_state_dict(average)
    model.eval()
    return model, preprocess


def predict(model, preprocess, values, device):
    output = []
    with torch.inference_mode():
        for offset in range(0, len(values), 2048):
            x = torch.from_numpy(normalize(values[offset:offset+2048], preprocess)).to(device)
            output.append(torch.softmax(model(x), 1).cpu().numpy())
    return np.concatenate(output)


def build_features(work, config, plan):
    cache = work/'cache'
    p = cache/'prepared.json';state = json.loads(p.read_text());assert verified_marker(p, state['fingerprint'])
    train, test = pd.read_parquet(cache/'train.parquet'), pd.read_parquet(cache/'test.parquet')
    assert len(train) == config['expected']['train'] and len(test) == config['expected']['test']
    assert 'label' not in test and not set(train.author) & set(test.author)
    assert train.groupby('author').fold.nunique().max() == 1
    posts = pd.read_parquet(cache/'all_posts.parquet')
    extra_path = work/plan['past_features']
    assert extra_path.resolve().is_relative_to(work.resolve())
    assert digest(extra_path) == plan['past_features_sha256']
    extra = pd.read_parquet(extra_path)
    numeric = json.loads((cache/'feature_columns.json').read_text())['numeric']
    extra_columns = [c for c in extra if c not in KEYS]
    assert all(c.startswith(('rh_', 'sr_')) for c in extra_columns)
    train = train.merge(extra, on=KEYS, validate='one_to_one')
    test = test.merge(extra, on=KEYS, validate='one_to_one')
    index = pd.MultiIndex.from_frame(posts[KEYS])
    assert index.is_unique
    train_index, test_index = (index.get_indexer(pd.MultiIndex.from_frame(df[KEYS])) for df in (train, test))
    assert min(train_index.min(), test_index.min()) >= 0
    blocks_train, blocks_test = [], []
    for role, filename in [('text_encoder', 'text_embeddings'), ('image_encoder', 'image_embeddings')]:
        fp = identity({'settings': config[role], 'data': digest(cache/'all_posts.parquet')})
        assert verified_marker(cache/f'{filename}.json', fp)
        values = np.load(cache/f'{filename}.npy', allow_pickle=False, mmap_mode='r')
        assert len(values) == len(posts)
        blocks_train.append(values[train_index]);blocks_test.append(values[test_index])
    for df, blocks in ((train, blocks_train), (test, blocks_test)):
        numeric_values = df[numeric+extra_columns].to_numpy(dtype=np.float32)
        blocks.append(numeric_values)
        blocks.append(np.isnan(numeric_values).astype(np.float32))
    xtrain, xtest = np.concatenate(blocks_train, axis=1), np.concatenate(blocks_test, axis=1)
    assert np.isfinite(xtrain[~np.isnan(xtrain)]).all() and np.isfinite(xtest[~np.isnan(xtest)]).all()
    return train, test, xtrain, xtest


def evaluate_candidate(spec, config, train, test, xtrain, xtest, work, group_fp, device, budget_end):
    fp = identity({'parent': group_fp, 'spec': spec})
    folder = work/'models'/spec['name'];folder.mkdir(parents=True, exist_ok=True)
    if verified_marker(folder/'complete.json', fp):
        return np.load(folder/'probabilities.npz', allow_pickle=False)
    oof = np.zeros((len(train), 3), np.float32)
    hard_end = datetime.fromisoformat(config['deadline']).timestamp() if config.get('deadline') else float('inf')
    for fold in sorted(train.fold.unique()):
        stop_check(work)
        valid = np.flatnonzero(train.fold.to_numpy() == fold)
        fit = np.flatnonzero(train.fold.to_numpy() != fold)
        fold_fp = identity({'parent': fp, 'fold': int(fold)})
        marker = folder/f'fold-{fold}.json';probs = folder/f'fold-{fold}.npz'
        if not verified_marker(marker, fold_fp):
            if time.time() >= budget_end:
                raise base.BudgetReached('Original search budget reached')
            atomic_json(work/'training-status.json', {'candidate': spec['name'], 'job': f'fold-{fold}', 'stage': 'fit', 'optimizer': 'pytorch', 'at': now()})
            started = time.monotonic()
            model, prep = fit_network(xtrain[fit], train.iloc[fit].label, spec, folder/'checkpoints'/f'fold-{fold}',
                                      work, fold_fp, device, config['seed'], hard_end)
            values = predict(model, prep, xtrain[valid], device);base.check_probabilities(values, len(valid))
            atomic_torch(folder/f'fold-{fold}.pt', {'weights': model.state_dict(), 'preprocess': prep, 'spec': spec})
            with probs.with_suffix('.tmp').open('wb') as stream:np.savez_compressed(stream, probabilities=values)
            probs.with_suffix('.tmp').replace(probs)
            mark(marker, fold_fp, [probs, folder/f'fold-{fold}.pt'], elapsed_seconds=time.monotonic()-started)
        oof[valid] = np.load(probs, allow_pickle=False)['probabilities']
    stop_check(work)
    atomic_json(work/'training-status.json', {'candidate': spec['name'], 'job': 'full', 'stage': 'full_fit', 'optimizer': 'pytorch', 'at': now()})
    model, prep = fit_network(xtrain, train.label, spec, folder/'checkpoints/full', work, fp, device, config['seed'], hard_end)
    values = predict(model, prep, xtest, device);base.check_probabilities(values, len(test))
    atomic_torch(folder/'model.pt', {'weights': model.state_dict(), 'preprocess': prep, 'spec': spec})
    probability_path = folder/'probabilities.npz'
    with probability_path.with_suffix('.tmp').open('wb') as stream:np.savez_compressed(stream, oof=oof, test=values)
    probability_path.with_suffix('.tmp').replace(probability_path)
    atomic_json(folder/'metrics.json', {'selection_split': 'train_author_oof', 'spec': spec, 'oof': base.score(train.label, oof.argmax(1)),
                'folds': [base.score(train.loc[train.fold == fold, 'label'], oof[train.fold == fold].argmax(1)) for fold in sorted(train.fold.unique())]})
    mark(folder/'complete.json', fp, [folder/'model.pt', probability_path, folder/'metrics.json'])
    return np.load(probability_path, allow_pickle=False)


def run(work, plan_path, group, helper_path, device='cuda'):
    stop_check(work)
    assert not (work/'report/official-test.json').exists()
    parent = json.loads((work/'environment/manifest.json').read_text());config = parent['config']
    assert parent['fingerprint'] == identity({'code': artifacts_fingerprint(Path(base.__file__).resolve().parents[1]), 'config': config})
    plan = json.loads(plan_path.read_text());specs = plan['candidates']
    assert 1 <= len(specs) <= 4 and len({s['name'] for s in specs}) == len(specs)
    assert all(s['name'].startswith(f'refine-{group}-') and s['family'] == 'neural' for s in specs)
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    if device == 'cuda':
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        assert torch.ones(1024, device='cuda').sum().item() == 1024
    helper_spec = importlib.util.spec_from_file_location('frozen_refinement_helper', helper_path)
    helper = importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
    experiment = work/'environment/refinements'/group;experiment.mkdir(parents=True, exist_ok=True)
    with (experiment/'runtime-history.jsonl').open('a') as stream:
        stream.write(json.dumps({'at': now(), 'device': device, 'torch': str(torch.__version__),
                     'gpu_uuid': os.environ.get('NVIDIA_VISIBLE_DEVICES') if device == 'cuda' else None,
                     'gpu_name': torch.cuda.get_device_name(0) if device == 'cuda' else None,
                     'single_gpu_cuda_reduction': 'passed' if device == 'cuda' else 'not-requested'})+'\n')
    inputs = {name: digest(work/'cache'/name) for name in (
        'train.parquet', 'test.parquet', 'prepared.json', 'text_embeddings.json', 'image_embeddings.json')}
    fp = identity({'parent': parent['fingerprint'], 'inputs': inputs, 'script': digest(Path(__file__)),
                   'helper': digest(helper_path), 'plan': plan, 'device_type': device})
    marker = experiment/'manifest.json'
    if marker.exists():assert json.loads(marker.read_text())['fingerprint'] == fp
    else:
        atomic_json(marker, {'at': now(), 'fingerprint': fp, 'scientific_extension': True, 'parent_fingerprint': parent['fingerprint'],
                    'script_sha256': digest(Path(__file__)), 'helper_sha256': digest(helper_path), 'plan': plan, 'inputs': inputs,
                    'device_type': device, 'dtype': 'float32', 'torch_version': str(torch.__version__),
                    'test_targets_used': False, 'author_folds_preserved': True,
                    'same_search_budget': json.loads((work/'search-budget.json').read_text())})
        shutil.copy2(work/'selection.json', experiment/'selection-before.json')
        shutil.copy2(work/'predictions/final.csv', experiment/'predictions-before.csv')
    train, test, xtrain, xtest = build_features(work, config, plan)
    atomic_json(experiment/'feature-layout.json', {'train_shape': list(xtrain.shape), 'test_shape': list(xtest.shape),
                'normalization': 'median/mean/std fit within each training fold; no PCA; frozen current text/images and strict-past scalars'})
    predictions = helper.load_predictions(work)
    budget_end = json.loads((work/'search-budget.json').read_text())['ends']
    trials = []
    for spec in specs:
        stop_check(work)
        assert not (work/'report/official-test.json').exists()
        if time.time() >= budget_end:break
        try:
            with evaluate_candidate(spec, config, train, test, xtrain, xtest, work, fp, device, budget_end) as values:
                predictions[spec['name']] = {k: values[k].copy() for k in ('oof', 'test')}
        except base.BudgetReached:break
        oof, _, scales = helper.publish_best(work, train, test, predictions)
        trials.append({'candidate': spec['name'], 'metrics': json.loads((work/'models'/spec['name']/'metrics.json').read_text())['oof'],
                       'selected_metrics': json.loads((work/'selection.json').read_text())['metrics']})
        atomic_json(experiment/'progress.json', {'at': now(), 'trials': trials, 'best_selection': json.loads((work/'selection.json').read_text())})
        if trials[-1]['selected_metrics']['macro_f1'] >= .70:break
    oof, _, scales = helper.publish_best(work, train, test, predictions)
    selected = json.loads((work/'selection.json').read_text())
    report = {'selection': selected, 'candidates': {n: json.loads((work/'models'/n/'metrics.json').read_text()) for n in predictions},
              'confusion_train_oof': confusion_matrix(train.label, (oof*scales).argmax(1), labels=[0, 1, 2]).tolist(),
              'per_class_train_oof': classification_report(train.label, (oof*scales).argmax(1), output_dict=True, zero_division=0),
              'scientific_extensions': [str(p.relative_to(work)) for p in sorted((work/'environment/refinements').glob('*/manifest.json'))]}
    report.update(base.diagnostics(train, (oof*scales).argmax(1), predictions['baseline']['oof'].argmax(1), config['bootstrap_samples'], config['seed']))
    atomic_json(work/'report/training.json', report)
    lines = ['# Kairos models and authorized refinements', '', 'Selection uses train-author OOF; official test remains unseen.', '',
             '| Candidate | OOF macro-F1 | QWK |', '| --- | --- | --- |']
    for name, result in report['candidates'].items():lines.append(f"| {name} | {result['oof']['macro_f1']:.4f} | {result['oof']['qwk']:.4f} |")
    lines += ['', 'Neural trials use full frozen text/image vectors plus verified strict-past features; preprocessing is fit within each training fold.',
              'Optimizer checkpoints are atomic and versioned after every epoch, including shuffle and dropout RNG state.',
              'All selection/blending uses reused OOF targets and can be optimistic; collection-time profile features are retrospective.']
    (work/'report/REPORT.md').write_text('\n'.join(lines)+'\n')
    atomic_json(experiment/'complete.json', {'at': now(), 'fingerprint': fp, 'trials': trials, 'best_selection': selected})
    return selected


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=Path('/work'));parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--group', required=True);parser.add_argument('--helper', type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.work/'status.json', {'state': 'running', 'stage': 'train', 'extension': args.group, 'at': now()})
    try:
        result = run(args.work, args.plan, args.group, args.helper)
        atomic_json(args.work/'status.json', {'state': 'completed', 'stage': 'refinement', 'extension': args.group, 'at': now(), 'selection': result})
    except Exception as exc:
        atomic_json(args.work/'status.json', {'state': 'needs_attention', 'stage': 'refinement', 'extension': args.group,
                    'at': now(), 'error_type': type(exc).__name__, 'reason': str(exc)})
        raise
