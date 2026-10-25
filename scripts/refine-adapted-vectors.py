#!/usr/bin/env python3
"""Fold-specific supervised text vectors, followed by temporal-history trees."""
import argparse
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

import kairos_lab.train as base
from kairos_lab.common import artifacts_fingerprint, atomic_json, digest, identity, mark, now, stop_check, verified_marker
from kairos_lab.dataset import KEYS, canonical_text
from kairos_lab.encoders import contextual_embeddings, model_path


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def source_records(work, plan, folds):
    previous = work / 'environment/refinements' / plan['source_group']
    manifest = json.loads((previous / 'manifest.json').read_text())
    assert manifest['fingerprint'] == plan['source_group_fingerprint']
    specs = manifest['plan']['candidates']
    selected = next(s for s in specs if s['name'] == plan['source_candidate'])
    parent_fp = identity({'parent': manifest['fingerprint'], 'spec': selected})
    folder = work / 'models' / selected['name'];records = {}
    for job in [f'fold-{k}' for k in range(folds)] + ['full']:
        marker = folder / ('complete.json' if job == 'full' else job + '.json')
        expected = parent_fp if job == 'full' else identity({'parent': parent_fp, 'fold': int(job.split('-')[1])})
        assert verified_marker(marker, expected), f'Missing source encoder for {job}; full weights are not an OOF fallback'
        path = folder / ('model.pt' if job == 'full' else job + '.pt')
        records[job] = {'path': str(path.relative_to(work)), 'sha256': digest(path), 'source_fingerprint': expected}
    assert selected['max_length'] == plan['max_length']
    return manifest, records


def initialize(work, plan_path, native_path, ft_path):
    stop_check(work);assert not (work / 'report/official-test.json').exists()
    parent = json.loads((work / 'environment/manifest.json').read_text());config = parent['config']
    assert parent['fingerprint'] == identity({'code': artifacts_fingerprint(Path(base.__file__).resolve().parents[1]), 'config': config})
    plan = json.loads(plan_path.read_text());group = plan['group']
    assert group.startswith('r') and len(group) == 3 and group[1:].isdigit()
    assert 1 <= len(plan['candidates']) <= 2
    assert all(s['name'].startswith(f'refine-{group}-') and s['family'] == 'multimodal' for s in plan['candidates'])
    old, records = source_records(work, plan, config['folds'])
    assert old['parent_fingerprint'] == parent['fingerprint']
    for name, sha in old['inputs'].items():
        assert digest(work / 'cache' / name) == sha
    assert digest(ft_path) == old['script_sha256']
    graph_group = json.loads((work / plan['graph_manifest']).read_text())
    assert graph_group['fingerprint'] == plan['graph_group_fingerprint']
    for role in ('past_features', 'graph_features'):
        assert digest(work / plan[role]) == plan[role + '_sha256']
    for name in ('prepared', 'image_embeddings'):
        marker = work / 'cache' / (name + '.json');state = json.loads(marker.read_text())
        assert verified_marker(marker, state['fingerprint'])
    inputs = {name: digest(work / 'cache' / name) for name in
              ('train.parquet', 'test.parquet', 'all_posts.parquet', 'history_outcomes.parquet',
               'prepared.json', 'feature_columns.json', 'label-audit.json', 'image_embeddings.json')}
    fp = identity({'parent': parent['fingerprint'], 'inputs': inputs, 'source_encoders': records,
                   'script': digest(Path(__file__)), 'native_helper': digest(native_path), 'ft_helper': digest(ft_path), 'plan': plan})
    experiment = work / 'environment/refinements' / group;experiment.mkdir(parents=True, exist_ok=True)
    marker = experiment / 'manifest.json'
    if marker.exists():
        assert json.loads(marker.read_text())['fingerprint'] == fp
    else:
        atomic_json(marker, {'at': now(), 'fingerprint': fp, 'parent_fingerprint': parent['fingerprint'],
                    'inputs': inputs, 'source_encoders': records, 'source_group_fingerprint': old['fingerprint'],
                    'script_sha256': digest(Path(__file__)), 'native_helper_sha256': digest(native_path),
                    'ft_helper_sha256': digest(ft_path), 'plan': plan, 'scientific_extension': True,
                    'encoder_validation_rule': 'fold-k only for OOF fold k; full only for final full-data fit',
                    'test_targets_used': False, 'author_folds_preserved': True,
                    'same_search_budget': json.loads((work / 'search-budget.json').read_text())})
    return config, plan, experiment, fp, records


def encoder_fp(group_fp, job, record):
    return identity({'parent': group_fp, 'job': job, 'weights': record['sha256'], 'policy': 'l2-mean-last-hidden-v1'})


def commit_array(path, values):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as stream:
        np.save(stream, values, allow_pickle=False);stream.flush();os.fsync(stream.fileno())
    tmp.replace(path)


def load_backbone(snapshot, state, width):
    import torch
    from transformers import AutoModel
    model = AutoModel.from_pretrained(snapshot, local_files_only=True, add_pooling_layer=False,
                                     attn_implementation='eager', torch_dtype=torch.float32)
    assert model.config.model_type == 'xlm-roberta' and model.config.hidden_size == width
    model.load_state_dict({k.removeprefix('backbone.'): v for k, v in state['weights'].items() if k.startswith('backbone.')})
    return model


def encode_chunks(work, folder, fp, rows, width, batch_size, chunk_rows, compute,
                  budget_end=float('inf'), until_chunks=None, job='encode', candidate='refine-r07-encoding'):
    """Each immutable chunk commits before progress; completed chunks survive a stop."""
    folder.mkdir(parents=True, exist_ok=True);marker = folder / 'complete.json'
    if verified_marker(marker, fp):
        return True
    assert chunk_rows % batch_size == 0
    ranges = [(start, min(rows, start + chunk_rows)) for start in range(0, rows, chunk_rows)]
    existing = {}
    for start, end in ranges:
        path = folder / f'chunk-{start:07d}.json';part_fp = identity({'parent': fp, 'start': start, 'end': end})
        if verified_marker(path, part_fp):
            existing[start] = end
    done_steps = sum(math.ceil((end-start) / batch_size) for start, end in existing.items())
    initial_steps = done_steps;started = time.monotonic();new_chunks = 0
    atomic_json(work / 'training-status.json', {'candidate': candidate, 'job': job, 'stage': 'adapted_encoding', 'optimizer': 'pytorch', 'at': now()})
    for start, end in ranges:
        if start in existing:
            continue
        stop_check(work)
        if time.time() >= budget_end:
            raise base.BudgetReached('Original budget reached; committed encoding chunks retained')
        values = np.asarray(compute(start, end), dtype=np.float32)
        assert values.shape == (end-start, width) and np.isfinite(values).all()
        path = folder / f'chunk-{start:07d}.npy';commit_array(path, values)
        mark(path.with_suffix('.json'), identity({'parent': fp, 'start': start, 'end': end}), [path], start=start, end=end)
        done_steps += math.ceil((end-start) / batch_size);new_chunks += 1
        atomic_json(work / 'neural-progress.json', {'candidate': candidate, 'job': job, 'iterations': done_steps,
                    'total_iterations': math.ceil(rows / batch_size), 'session_iterations': done_steps-initial_steps,
                    'session_elapsed_seconds': time.monotonic()-started, 'at': now()})
        if until_chunks is not None and new_chunks >= until_chunks:
            return False
    path = folder / 'vectors.npy';tmp = folder / 'vectors.npy.tmp'
    array = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float32, shape=(rows, width))
    for start, end in ranges:
        part = folder / f'chunk-{start:07d}.npy'
        assert verified_marker(part.with_suffix('.json'), identity({'parent': fp, 'start': start, 'end': end}))
        array[start:end] = np.load(part, allow_pickle=False)
    array.flush();del array
    with tmp.open('rb') as stream:os.fsync(stream.fileno())
    tmp.replace(path)
    mark(marker, fp, [path], rows=rows, width=width)
    return True


def encode(work, models, config, plan, experiment, group_fp, records, ft_path, gpu_uuid):
    import torch
    from transformers import AutoTokenizer
    assert gpu_uuid and gpu_uuid.startswith('GPU-')
    torch.set_num_threads(config['threads']);torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
    assert torch.ones(1024, device='cuda').sum().item() == 1024
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    snapshot = model_path(models, 'text_encoder', config['text_encoder'])
    old = json.loads((work / 'environment/refinements' / plan['source_group'] / 'manifest.json').read_text())
    assert digest(snapshot / 'verified.json') == old['snapshot_marker_sha256']
    ft = module(ft_path, 'adapted_original_finetune')
    posts = pd.read_parquet(work / 'cache/all_posts.parquet')
    token_fp = identity({'parent': group_fp, 'policy': 'original-r05-body-document-head-tail'})
    if not verified_marker(experiment / 'tokens.json', token_fp):
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        texts = posts.text.fillna('').map(canonical_text) + ' document ' + posts.document_text.fillna('').str.slice(0, 2000).map(canonical_text)
        ids = np.zeros((len(posts), plan['max_length']), np.int32);masks = np.zeros_like(ids, dtype=np.int8)
        for offset in range(0, len(posts), 256):
            stop_check(work)
            tokens = tokenizer(texts.iloc[offset:offset+256].tolist(), add_special_tokens=False,
                               return_attention_mask=False, return_token_type_ids=False, truncation=False, verbose=False)['input_ids']
            for j, values in enumerate(tokens):
                ids[offset+j], masks[offset+j] = ft.pack_tokens(values, tokenizer, plan['max_length'])
        commit_array(experiment / 'ids.npy', ids);commit_array(experiment / 'masks.npy', masks)
        mark(experiment / 'tokens.json', token_fp, [experiment / 'ids.npy', experiment / 'masks.npy'], rows=len(posts))
    ids = np.load(experiment / 'ids.npy', mmap_mode='r');masks = np.load(experiment / 'masks.npy', mmap_mode='r')
    with (experiment / 'runtime-history.jsonl').open('a') as stream:
        stream.write(json.dumps({'at': now(), 'gpu_uuid_from_launcher': gpu_uuid, 'gpu_name': torch.cuda.get_device_name(0),
                                'torch': str(torch.__version__), 'batch_size': plan['batch_size'], 'cuda_check': 'passed'}) + '\n')
    budget_end = json.loads((work / 'search-budget.json').read_text())['ends']
    for job, record in records.items():
        stop_check(work);fp = encoder_fp(group_fp, job, record);folder = experiment / 'encoders' / job
        if verified_marker(folder / 'complete.json', fp):
            continue
        assert digest(work / record['path']) == record['sha256']
        state = torch.load(work / record['path'], map_location='cpu', weights_only=False)
        model = load_backbone(snapshot, state, plan['encoder_width'])
        del state;model.requires_grad_(False);model.eval().to('cuda')
        def compute(first, last):
            output = []
            with torch.inference_mode():
                for start in range(first, last, plan['batch_size']):
                    stop_check(work);end = min(last, start + plan['batch_size'])
                    length = int(masks[start:end].sum(1).max())
                    token_ids = torch.from_numpy(np.asarray(ids[start:end, :length], dtype=np.int64)).to('cuda')
                    attention = torch.from_numpy(np.asarray(masks[start:end, :length], dtype=np.int64)).to('cuda')
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        hidden = model(input_ids=token_ids, attention_mask=attention).last_hidden_state.float()
                    weights = attention.unsqueeze(-1).float();pooled = (hidden*weights).sum(1)/weights.sum(1).clamp(min=1)
                    output.append(torch.nn.functional.normalize(pooled, dim=1).cpu().numpy())
            return np.concatenate(output)
        encode_chunks(work, folder, fp, len(posts), plan['encoder_width'], plan['batch_size'], plan['chunk_rows'], compute,
                      budget_end=budget_end, job='encode-' + job, candidate=f"refine-{plan['group']}-encoding")
        del model;torch.cuda.empty_cache()
    mark(experiment / 'encoding-complete.json', group_fp,
         [experiment / 'encoders' / job / 'complete.json' for job in records])


def frames(work, plan, experiment, group_fp, records, job):
    assert job in records, 'No fallback to full encoder for an OOF job'
    folder = experiment / 'encoders' / job
    assert verified_marker(folder / 'complete.json', encoder_fp(group_fp, job, records[job]))
    posts = pd.read_parquet(work / 'cache/all_posts.parquet');outcomes = pd.read_parquet(work / 'cache/history_outcomes.parquet')
    assert posts[KEYS].equals(outcomes[KEYS])
    values = np.load(folder / 'vectors.npy', mmap_mode='r');assert values.shape == (len(posts), plan['encoder_width'])
    context_fp = identity({'parent': encoder_fp(group_fp, job, records[job]), 'recipe': 'original-strict20-context-with-own-outcomes'})
    if not verified_marker(folder / 'context.json', context_fp):
        context = contextual_embeddings(posts, values, outcomes.reactions.to_numpy(), work)
        commit_array(folder / 'context.npy', context)
        mark(folder / 'context.json', context_fp, [folder / 'context.npy'])
    context = np.load(folder / 'context.npy', mmap_mode='r')
    image_fp = identity({'settings': json.loads((work / 'environment/manifest.json').read_text())['config']['image_encoder'],
                         'data': digest(work / 'cache/all_posts.parquet')})
    assert verified_marker(work / 'cache/image_embeddings.json', image_fp)
    images = np.load(work / 'cache/image_embeddings.npy', mmap_mode='r')
    numeric = json.loads((work / 'cache/feature_columns.json').read_text())['numeric']
    tables = []
    for role in ('past_features', 'graph_features'):
        assert digest(work / plan[role]) == plan[role + '_sha256']
        table = pd.read_parquet(work / plan[role]);tables.append(table);numeric += [c for c in table if c not in KEYS]
    aux = pd.DataFrame(context[:, -8:], columns=[f'adapted_aux_{k}' for k in range(8)])
    aux[KEYS] = posts[KEYS];tables.append(aux);numeric += [f'adapted_aux_{k}' for k in range(8)]
    emb = np.concatenate([context[:, :-8], images], axis=1)
    emb_columns = [f'adapted_component_{k}' for k in range(context.shape[1]-8)] + [f'image_encoder_component_{k}' for k in range(images.shape[1])]
    embeddings = pd.DataFrame(emb, columns=emb_columns);embeddings[KEYS] = posts[KEYS];tables.append(embeddings)
    result = []
    for name in ('train', 'test'):
        frame = pd.read_parquet(work / 'cache' / (name + '.parquet'))
        original = frame[KEYS].copy()
        for table in tables:
            frame = frame.merge(table, on=KEYS, validate='one_to_one', sort=False)
        assert frame[KEYS].equals(original)
        result.append(frame)
    assert 'label' not in result[1] and not set(result[0].author) & set(result[1].author)
    assert result[0].groupby('author').fold.nunique().max() == 1
    return result[0], result[1], numeric, emb_columns


def evaluate(work, config, plan, experiment, group_fp, records, trial, helper, budget_end):
    folder = work / 'models' / trial['name'];folder.mkdir(parents=True, exist_ok=True)
    fp = identity({'parent': group_fp, 'spec': trial,
                   'encoder_jobs': {job: encoder_fp(group_fp, job, record) for job, record in records.items()}})
    if verified_marker(folder / 'complete.json', fp):
        return np.load(folder / 'probabilities.npz', allow_pickle=False)
    target = pd.read_parquet(work / 'cache/train.parquet', columns=['fold', 'label']);oof = np.zeros((len(target), 3))
    for fold in sorted(target.fold.unique()):
        job = f'fold-{int(fold)}';fold_fp = identity({'parent': fp, 'fold': int(fold)});marker = folder / (job + '.json')
        stop_check(work);assert not (work / 'report/official-test.json').exists()
        if verified_marker(marker, fold_fp):
            with np.load(folder / (job + '.npz'), allow_pickle=False) as z:oof[target.fold == fold] = z['probabilities']
            continue
        if time.time() >= budget_end:raise base.BudgetReached('Original search budget reached')
        atomic_json(work / 'training-status.json', {'candidate': trial['name'], 'job': 'preparation-' + job, 'stage': 'feature_preparation'})
        train, _, numeric, embeddings = frames(work, plan, experiment, group_fp, records, job)
        assert train[['fold', 'label']].equals(target)
        fit = np.flatnonzero(train.fold.to_numpy() != fold);valid = np.flatnonzero(train.fold.to_numpy() == fold)
        atomic_json(work / 'training-status.json', {'candidate': trial['name'], 'fold': int(fold), 'stage': 'fit'})
        started = time.monotonic();model = helper.extended_estimator(trial, config, numeric, train.iloc[fit].label, embeddings)
        model = base.fit_checkpointed(model, trial, train.iloc[fit], train.iloc[fit].label, folder, job, work, fold_fp)
        values = model.predict_proba(train.iloc[valid]);base.check_probabilities(values, len(valid));oof[valid] = values
        path = folder / (job + '.npz')
        with path.with_suffix('.tmp').open('wb') as stream:np.savez_compressed(stream, probabilities=values)
        path.with_suffix('.tmp').replace(path);base.atomic_model(folder / (job + '.joblib'), model)
        mark(marker, fold_fp, [path, folder / (job + '.joblib')], elapsed_seconds=time.monotonic()-started,
             encoder_job=job, source_weight_sha256=records[job]['sha256'])
        del train, model
    stop_check(work);assert not (work / 'report/official-test.json').exists()
    atomic_json(work / 'training-status.json', {'candidate': trial['name'], 'job': 'preparation-full', 'stage': 'feature_preparation'})
    train, test, numeric, embeddings = frames(work, plan, experiment, group_fp, records, 'full')
    atomic_json(work / 'training-status.json', {'candidate': trial['name'], 'stage': 'full_fit'})
    model = helper.extended_estimator(trial, config, numeric, train.label, embeddings)
    model = base.fit_checkpointed(model, trial, train, train.label, folder, 'full', work, fp)
    values = model.predict_proba(test);base.check_probabilities(values, len(test));base.atomic_model(folder / 'model.joblib', model)
    path = folder / 'probabilities.npz'
    with path.with_suffix('.tmp').open('wb') as stream:np.savez_compressed(stream, oof=oof, test=values)
    path.with_suffix('.tmp').replace(path)
    atomic_json(folder / 'metrics.json', {'selection_split': 'train_author_oof', 'spec': trial, 'oof': base.score(train.label, oof.argmax(1)),
                'folds': [base.score(train.loc[train.fold == k, 'label'], oof[train.fold == k].argmax(1)) for k in sorted(train.fold.unique())],
                'encoder_validation_rule': 'fold-k for fold k; full only for full-data fit'})
    mark(folder / 'complete.json', fp, [folder / 'model.joblib', path, folder / 'metrics.json'])
    return np.load(path, allow_pickle=False)


def train(work, config, plan, experiment, group_fp, records, native_path):
    helper = module(native_path, 'adapted_tree_helper')
    assert verified_marker(experiment / 'encoding-complete.json', group_fp)
    predictions = helper.load_predictions(work);trials = [];end = json.loads((work / 'search-budget.json').read_text())['ends']
    for trial in plan['candidates']:
        stop_check(work);assert not (work / 'report/official-test.json').exists()
        if time.time() >= end:break
        try:
            with evaluate(work, config, plan, experiment, group_fp, records, trial, helper, end) as values:
                predictions[trial['name']] = {k: values[k].copy() for k in ('oof', 'test')}
        except base.BudgetReached:
            break
        train_frame = pd.read_parquet(work / 'cache/train.parquet');test_frame = pd.read_parquet(work / 'cache/test.parquet')
        helper.publish_best(work, train_frame, test_frame, predictions)
        trials.append({'candidate': trial['name'], 'metrics': json.loads((work / 'models' / trial['name'] / 'metrics.json').read_text())['oof'],
                       'selected_metrics': json.loads((work / 'selection.json').read_text())['metrics']})
        atomic_json(experiment / 'progress.json', {'at': now(), 'trials': trials})
        if trials[-1]['selected_metrics']['macro_f1'] >= .70:break
    train_frame = pd.read_parquet(work / 'cache/train.parquet');test_frame = pd.read_parquet(work / 'cache/test.parquet')
    oof, _, scales = helper.publish_best(work, train_frame, test_frame, predictions);selected = json.loads((work / 'selection.json').read_text())
    report = {'selection': selected, 'candidates': {n: json.loads((work / 'models' / n / 'metrics.json').read_text()) for n in predictions},
              'confusion_train_oof': confusion_matrix(train_frame.label, (oof*scales).argmax(1), labels=[0,1,2]).tolist(),
              'per_class_train_oof': classification_report(train_frame.label, (oof*scales).argmax(1), output_dict=True, zero_division=0),
              'scientific_extensions': [str(p.relative_to(work)) for p in sorted((work / 'environment/refinements').glob('*/manifest.json'))]}
    report.update(base.diagnostics(train_frame, (oof*scales).argmax(1), predictions['baseline']['oof'].argmax(1), config['bootstrap_samples'], config['seed']))
    atomic_json(work / 'report/training.json', report)
    lines = ['# Kairos: fold-specific adapted text vectors and history trees', '', 'Selection uses train-author OOF only; official test remains unseen.', '',
             '| Candidate | OOF macro-F1 | QWK |', '| --- | --- | --- |']
    for name, result in report['candidates'].items():lines.append(f"| {name} | {result['oof']['macro_f1']:.4f} | {result['oof']['qwk']:.4f} |")
    lines += ['', 'OOF fold k uses only r05 fold-k encoder weights. Full-label-trained weights are used only for the final full-data fit.',
              'L2-normalized supervised text vectors feed strict past-only author context and trees; source encoders are not retrained.',
              'Graph features retain the declared earlier-unlabeled transductive structure; no cross-author outcomes or test targets.',
              'OOF is reused for selection and can be optimistic; all source weights/input/spec hashes are preserved.']
    (work / 'report/REPORT.md').write_text('\n'.join(lines) + '\n')
    atomic_json(experiment / 'complete.json', {'at': now(), 'fingerprint': group_fp, 'trials': trials, 'best_selection': selected})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('stage', choices=['encode', 'train'])
    parser.add_argument('--work', type=Path, default=Path('/work'));parser.add_argument('--models', type=Path, default=Path('/models'))
    parser.add_argument('--plan', type=Path, required=True);parser.add_argument('--native-helper', type=Path, required=True)
    parser.add_argument('--ft-helper', type=Path, required=True);parser.add_argument('--gpu-uuid')
    args = parser.parse_args();plan = json.loads(args.plan.read_text())
    atomic_json(args.work / 'status.json', {'state': 'running', 'stage': 'train', 'extension': plan['group'], 'at': now()})
    atomic_json(args.work / 'training-status.json', {'candidate': f"refine-{plan['group']}-preparation", 'job': args.stage + '-preparation', 'stage': 'preparation'})
    try:
        config, plan, experiment, fp, records = initialize(args.work, args.plan, args.native_helper, args.ft_helper)
        if args.stage == 'encode':encode(args.work, args.models, config, plan, experiment, fp, records, args.ft_helper, args.gpu_uuid)
        else:train(args.work, config, plan, experiment, fp, records, args.native_helper)
        atomic_json(args.work / 'status.json', {'state': 'completed', 'stage': 'refinement', 'extension': plan['group'], 'substage': args.stage, 'at': now()})
    except base.BudgetReached:
        atomic_json(args.work / 'status.json', {'state': 'completed', 'stage': 'refinement', 'extension': plan['group'], 'reason': 'original budget reached; previous selection retained', 'at': now()})
    except Exception as exc:
        atomic_json(args.work / 'status.json', {'state': 'needs_attention', 'stage': 'refinement', 'extension': plan['group'], 'error_type': type(exc).__name__, 'reason': str(exc), 'at': now()})
        raise
