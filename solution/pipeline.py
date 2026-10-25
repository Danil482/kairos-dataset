"""Rebuild features and predict with the fixed, already selected ensemble."""
from __future__ import annotations

import gc
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from kairos_lab import dataset, encoders, train as base
from kairos_lab.common import atomic_json, digest, identity, mark, stop_check, verified_marker
from solution.frozen import adapted, finetune, graph, history, nn_utils

ROOT = Path(__file__).resolve().parent
FUSION = 'refine-r08-text-adapt-last12-e3'
TEACHER = 'refine-r05-text-adapt-last4'
TREE = 'refine-r07-adapted-history128'


def settings():
    value = json.loads((ROOT / 'config.json').read_text())
    for name, sha in value['frozen_source_sha256'].items():
        if digest(ROOT.parent / name) != sha:
            raise ValueError(f'Frozen scientific helper changed: {name}')
    return value


def configure(device, threads):
    torch.set_num_threads(threads)
    if device == 'cuda' and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError('CUDA inference requires exactly one visible GPU')


def prepare(data, work, cfg):
    work.mkdir(parents=True, exist_ok=True)
    stop_check(work)
    fp = identity({'release': dataset.release_identity(data), 'settings': cfg})
    path = work / 'reproduction-input.json'
    if path.exists() and json.loads(path.read_text())['fingerprint'] != fp:
        raise ValueError('Data or fixed configuration changed; use a new work directory')
    atomic_json(path, {'fingerprint': fp, 'selection': 'fixed; no model search or test scoring'})
    print('Preparing official release and past-only features', flush=True)
    dataset.prepare(data, work, cfg['config'])


def raw_embeddings(data, weights, work, cfg, device):
    # Preserve the original float32 frozen-encoder policy before supervised BF16 inference.
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    for role in ('text_encoder', 'image_encoder'):
        encoders.model_path(weights / 'pretrained', role, cfg['config'][role])
    print('Frozen text and image encoders', flush=True)
    encoders.encode_text(weights / 'pretrained', work, cfg['config'], device)
    encoders.encode_images(data, weights / 'pretrained', work, cfg['config'], device)


def past_tables(data, work, cfg):
    cache = work / 'cache'
    inputs = {name: digest(cache / name) for name in
              ('all_posts.parquet', 'history_outcomes.parquet', 'text_embeddings.npy', 'image_embeddings.npy')}
    fp = identity({'inputs': inputs, 'frozen': cfg['frozen_source_sha256'],
                   'graph_cutoff_policy': 'original-r06-float64-train-residuals'})
    folder = cache / 'history'
    folder.mkdir(exist_ok=True)
    marker = folder / 'complete.json'
    if verified_marker(marker, fp):
        return
    print('Strictly earlier author history and entity graph', flush=True)
    # This is the unchanged R02 feature builder, including the frozen-text context scalars.
    frames = history.build_frames(work, cfg['config'], folder)
    del frames
    gc.collect()
    posts = pd.read_parquet(cache / 'all_posts.parquet')
    outcomes = pd.read_parquet(cache / 'history_outcomes.parquet')
    # R06 deliberately derived its own float64 train residual cutoffs. The label
    # reproduction audit can use extended precision; its cutoffs are not a substitute.
    from kairos.label import fit_cutoffs
    authors = set(pd.read_parquet(cache / 'train.parquet', columns=['author']).author)
    residual = pd.Series(np.log1p(np.maximum(outcomes.reactions.to_numpy(float), 0)) - posts.level)
    cutoffs = fit_cutoffs(residual[posts.author.isin(authors) & posts.history_count.ge(5)])
    raw, _ = graph.load_raw_posts(data, work)
    graph.graph_features(posts, outcomes, raw, cutoffs, work).to_parquet(cache / 'graph.parquet', index=False)
    mark(marker, fp, [folder / 'past-only-features.parquet', cache / 'graph.parquet'])


def token_arrays(frame, snapshot, length, work):
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    texts = (frame.text.fillna('').map(dataset.canonical_text) + ' document '
             + frame.document_text.fillna('').str.slice(0, 2000).map(dataset.canonical_text))
    ids = np.zeros((len(frame), length), np.int32)
    masks = np.zeros_like(ids, dtype=np.int8)
    for start in range(0, len(frame), 256):
        stop_check(work)
        tokens = tokenizer(texts.iloc[start:start+256].tolist(), add_special_tokens=False,
                           return_attention_mask=False, return_token_type_ids=False,
                           truncation=False, verbose=False)['input_ids']
        for i, values in enumerate(tokens):
            ids[start+i], masks[start+i] = finetune.pack_tokens(values, tokenizer, length)
    return ids, masks


def adapted_vectors(weights, work, cfg, device, deadline=float('inf')):
    cache = work / 'cache'
    weight_file = weights / 'models' / TEACHER / 'model.pt'
    fp = identity({'weights': digest(weight_file), 'posts': digest(cache / 'all_posts.parquet'),
                   'source': cfg['frozen_source_sha256'], 'device_policy': device})
    folder = cache / 'adapted'
    if verified_marker(folder / 'complete.json', fp):
        return
    print('R05 full-data backbone; independent fresh encoding', flush=True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == 'cuda' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('The frozen supervised inference policy requires BF16 support')
    snapshot = encoders.model_path(weights / 'pretrained', 'text_encoder', cfg['config']['text_encoder'])
    posts = pd.read_parquet(cache / 'all_posts.parquet')
    ids, masks = token_arrays(posts, snapshot, cfg['specs'][TEACHER]['max_length'], work)
    state = torch.load(weight_file, map_location='cpu', weights_only=False)
    assert state['spec'] == cfg['specs'][TEACHER]
    width = json.loads((snapshot / 'config.json').read_text())['hidden_size']
    model = adapted.load_backbone(snapshot, state, width)
    del state
    model.requires_grad_(False).eval().to(device)
    def compute(first, last):
        values = []
        with torch.inference_mode():
            for start in range(first, last, 32):
                stop_check(work)
                end = min(last, start+32)
                length = int(masks[start:end].sum(1).max())
                tokens = torch.from_numpy(np.asarray(ids[start:end, :length], dtype=np.int64)).to(device)
                attention = torch.from_numpy(np.asarray(masks[start:end, :length], dtype=np.int64)).to(device)
                precision = torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else nullcontext()
                with precision:
                    hidden = model(input_ids=tokens, attention_mask=attention).last_hidden_state.float()
                mask = attention.unsqueeze(-1).float()
                pooled = (hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
                values.append(torch.nn.functional.normalize(pooled, dim=1).cpu().numpy())
        return np.concatenate(values)
    adapted.encode_chunks(work, folder, fp, len(posts), width, 32, 2048, compute,
                          budget_end=deadline, job='full-inference', candidate='fixed-r05-backbone')
    del model
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()


def tree_frames(work):
    cache = work / 'cache'
    posts = pd.read_parquet(cache / 'all_posts.parquet')
    outcomes = pd.read_parquet(cache / 'history_outcomes.parquet')
    assert posts[dataset.KEYS].equals(outcomes[dataset.KEYS])
    values = np.load(cache / 'adapted/vectors.npy', mmap_mode='r')
    context = encoders.contextual_embeddings(posts, values, outcomes.reactions.to_numpy(), work)
    images = np.load(cache / 'image_embeddings.npy', mmap_mode='r')
    numeric = json.loads((cache / 'feature_columns.json').read_text())['numeric']
    tables = []
    for path in (cache / 'history/past-only-features.parquet', cache / 'graph.parquet'):
        table = pd.read_parquet(path)
        tables.append(table)
        numeric += [c for c in table if c not in dataset.KEYS]
    aux = pd.DataFrame(context[:, -8:], columns=[f'adapted_aux_{i}' for i in range(8)])
    aux[dataset.KEYS] = posts[dataset.KEYS]
    tables.append(aux)
    numeric += [f'adapted_aux_{i}' for i in range(8)]
    columns = ([f'adapted_component_{i}' for i in range(context.shape[1]-8)]
               + [f'image_encoder_component_{i}' for i in range(images.shape[1])])
    table = pd.DataFrame(np.concatenate([context[:, :-8], images], axis=1), columns=columns)
    table[dataset.KEYS] = posts[dataset.KEYS]
    tables.append(table)
    frames = []
    for name in ('train', 'test'):
        frame = pd.read_parquet(cache / (name+'.parquet'))
        keys = frame[dataset.KEYS].copy()
        for table in tables:
            frame = frame.merge(table, on=dataset.KEYS, validate='one_to_one', sort=False)
        assert frame[dataset.KEYS].equals(keys)
        frames.append(frame)
    return *frames, numeric, columns


def fusion_arrays(work, weights, cfg, partition):
    cache = work / 'cache'
    frame = pd.read_parquet(cache / (partition+'.parquet'))
    extra = pd.read_parquet(cache / 'history/past-only-features.parquet')
    names = [c for c in extra if c not in dataset.KEYS]
    columns = json.loads((cache / 'feature_columns.json').read_text())['numeric']
    merged = frame.merge(extra, on=dataset.KEYS, validate='one_to_one')
    posts = pd.read_parquet(cache / 'all_posts.parquet', columns=dataset.KEYS)
    indexes = pd.MultiIndex.from_frame(posts).get_indexer(pd.MultiIndex.from_frame(frame[dataset.KEYS]))
    assert (indexes >= 0).all()
    images = np.load(cache / 'image_embeddings.npy', mmap_mode='r')
    values = merged[columns+names].to_numpy(dtype=np.float32)
    numeric = np.concatenate([values, np.isnan(values).astype(np.float32), images[indexes]], axis=1)
    snapshot = encoders.model_path(weights / 'pretrained', 'text_encoder', cfg['config']['text_encoder'])
    ids, masks = token_arrays(frame, snapshot, cfg['specs'][FUSION]['max_length'], work)
    return frame, (ids, masks, numeric), snapshot


def verify_weights(weights, cfg, pretrained_only=False):
    if pretrained_only:
        for role in ('text_encoder', 'image_encoder'):
            encoders.model_path(weights / 'pretrained', role, cfg['config'][role])
        return
    manifest = json.loads((weights / 'manifest.json').read_text())
    if manifest['configuration_sha256'] != identity(cfg):
        raise ValueError('Weights belong to a different frozen solution')
    for name, sha in manifest['files'].items():
        path = weights / name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink():
            raise ValueError('Unsafe weight path')
        if not path.is_file() or digest(path) != sha:
            raise ValueError(f'Missing or modified weight: {name}')
    required = json.loads((ROOT / 'reference-weights.json').read_text())['files']
    if not required.keys() <= manifest['files'].keys():
        raise ValueError('Incomplete ensemble weights')
    if manifest['origin'] == 'original-frozen-run':
        for name, sha in required.items():
            if manifest['files'][name] != sha:
                raise ValueError(f'Original weight identity mismatch: {name}')
    elif manifest['origin'] != 'fixed-full-training':
        raise ValueError('Unrecognised weight provenance')
    verify_weights(weights, cfg, pretrained_only=True)


def weight_manifest(weights, cfg, origin):
    paths = [p for folder in ('models', 'pretrained') for p in (weights / folder).rglob('*')
             if p.is_file() and p.name != '.DS_Store']
    value = {'format_version': 1, 'origin': origin, 'configuration_sha256': identity(cfg),
             'files': {str(p.relative_to(weights)): digest(p) for p in sorted(paths)}}
    atomic_json(weights / 'manifest.json', value)
    return value


def predict(data, weights, work, output, cfg, device, check_reference=False, deadline=float('inf')):
    started = time.monotonic()
    configure(device, cfg['config']['threads'])
    verify_weights(weights, cfg)
    prepare(data, work, cfg)
    raw_embeddings(data, weights, work, cfg, device)
    past_tables(data, work, cfg)
    adapted_vectors(weights, work, cfg, device, deadline)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    frame, arrays, snapshot = fusion_arrays(work, weights, cfg, 'test')
    state = torch.load(weights / 'models' / FUSION / 'model.pt', map_location='cpu', weights_only=False)
    assert state['spec'] == cfg['specs'][FUSION]
    model = finetune.TextFusion(snapshot, arrays[2].shape[1], state['spec'])
    model.load_state_dict(state['weights'])
    model.eval().to(device)
    print('Frozen R08 fusion head: test forward pass', flush=True)
    fusion = finetune.predict_encoder(model, state['preprocess'], *arrays, device, state['spec'], nn_utils)
    del model, state, arrays
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    print('Frozen tree models and fixed blend', flush=True)
    train, tree_test, _, _ = tree_frames(work)
    del train
    baseline = pd.read_parquet(work / 'cache/baseline.parquet').drop(columns='split')
    baseline_test = frame[dataset.KEYS].merge(baseline, on=dataset.KEYS, validate='one_to_one')
    frames = {'baseline': baseline_test, 'history-1': frame, TREE: tree_test}
    probabilities = {FUSION: fusion}
    for name, table in frames.items():
        probabilities[name] = joblib.load(weights / 'models' / name / 'model.joblib').predict_proba(table)
    for values in probabilities.values():
        base.check_probabilities(values, len(frame))
    blend = sum(weight*probabilities[name] for name, weight in cfg['weights'].items())
    base.write_prediction(output, frame, (blend*np.asarray(cfg['class_scales'])).argmax(1))
    np.savez_compressed(work / 'component-probabilities.npz', **probabilities)
    sha = digest(output)
    expected = cfg['reference_prediction_sha256']
    result = {'rows': len(frame), 'prediction_sha256': sha, 'reference_prediction_sha256': expected,
              'reference_match': sha == expected, 'fresh_model_inference': True,
              'cached_prediction_files_read': False, 'elapsed_seconds': time.monotonic()-started,
              'device': device, 'torch': str(torch.__version__),
              'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
              'configuration_sha256': identity(cfg), 'weights_manifest_sha256': digest(weights / 'manifest.json'),
              'feature_sha256': {p.name: digest(p) for p in
                                (work / 'cache/all_posts.parquet', work / 'cache/text_embeddings.npy',
                                 work / 'cache/image_embeddings.npy', work / 'cache/graph.parquet',
                                 work / 'cache/history/past-only-features.parquet', work / 'cache/adapted/vectors.npy')}}
    if check_reference:
        expected_features = json.loads((ROOT / 'reference-features.json').read_text())
        result['feature_reference_match'] = result['feature_sha256'] == expected_features
    atomic_json(work / 'verification.json', result)
    print(json.dumps(result), flush=True)
    if check_reference and not result['reference_match']:
        raise ValueError('Fresh inference differs from the frozen reference CSV; no model selection performed')
    if check_reference and not result['feature_reference_match']:
        raise ValueError('Intermediate features differ from the frozen reference; use the pinned runtime')
    return result


def train(data, weights, work, cfg, device, deadline=float('inf')):
    """Fit only the five fixed full-data models; never search or score test targets."""
    configure(device, cfg['config']['threads'])
    verify_weights(weights, cfg, pretrained_only=True)
    if (weights / 'models').exists():
        raise ValueError('Use an output weights directory without existing trained models')
    prepare(data, work, cfg)
    raw_embeddings(data, weights, work, cfg, device)
    past_tables(data, work, cfg)
    frame = pd.read_parquet(work / 'cache/train.parquet')
    baseline = pd.read_parquet(work / 'cache/baseline.parquet').drop(columns='split')
    baseline_train = frame[dataset.KEYS+['label']].merge(baseline, on=dataset.KEYS, validate='one_to_one')
    numeric = json.loads((work / 'cache/feature_columns.json').read_text())['numeric']
    for name, rows in [('baseline', baseline_train), ('history-1', frame)]:
        spec = cfg['specs'][name]
        model = base.estimator(spec, cfg['config'], dataset.BASE_NUM if name == 'baseline' else numeric, rows.label)
        model = base.fit_checkpointed(model, spec, rows, rows.label, work / 'training' / name,
                                      'full', work, identity({'settings': cfg, 'name': name}))
        folder = weights / 'models' / name
        folder.mkdir(parents=True)
        base.atomic_model(folder / 'model.joblib', model)
    for name in (TEACHER, FUSION):
        spec = cfg['specs'][name]
        train_frame, arrays, snapshot = fusion_arrays(work, weights, cfg, 'train')
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model, prep = finetune.fit_encoder(*arrays, train_frame.label, snapshot, spec,
                         work / 'training' / name, work, identity({'settings': cfg, 'name': name}),
                         device, cfg['config']['seed'], deadline, nn_utils)
        folder = weights / 'models' / name
        folder.mkdir(parents=True)
        nn_utils.atomic_torch(folder / 'model.pt', {'weights': model.state_dict(), 'preprocess': prep, 'spec': spec})
        del model, arrays
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        if name == TEACHER:
            adapted_vectors(weights, work, cfg, device, deadline)
            train_frame, _, numeric, columns = tree_frames(work)
            model = history.extended_estimator(cfg['specs'][TREE], cfg['config'], numeric, train_frame.label, columns)
            model = base.fit_checkpointed(model, cfg['specs'][TREE], train_frame, train_frame.label,
                                          work / 'training' / TREE, 'full', work,
                                          identity({'settings': cfg, 'name': TREE}))
            folder = weights / 'models' / TREE
            folder.mkdir(parents=True)
            base.atomic_model(folder / 'model.joblib', model)
            del model, train_frame
            gc.collect()
    weight_manifest(weights, cfg, 'fixed-full-training')
    print('Fixed full-data training complete; test targets were not scored', flush=True)
