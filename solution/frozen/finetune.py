#!/usr/bin/env python3
"""Bounded supervised adaptation of the pinned local text encoder, author OOF only."""
import argparse
import gc
import importlib.util
import json
import math
import os
import re
import shutil
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix

import kairos_lab.train as base
from kairos_lab.common import artifacts_fingerprint, atomic_json, digest, identity, mark, now, stop_check, verified_marker
from kairos_lab.dataset import KEYS, canonical_text
from kairos_lab.encoders import model_path, save_array


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def pack_tokens(tokens, tokenizer, length):
    """XLM-R sequence layout: short single input, or first/tail pair within the cap."""
    if len(tokens) <= length-2:
        result = [tokenizer.cls_token_id]+tokens+[tokenizer.sep_token_id]
    else:
        first = (length-4)//2
        result = [tokenizer.cls_token_id]+tokens[:first]+[tokenizer.sep_token_id]*2+tokens[-(length-4-first):]+[tokenizer.sep_token_id]
    assert len(result) <= length
    ids = np.full(length, tokenizer.pad_token_id, dtype=np.int32)
    mask = np.zeros(length, dtype=np.int8)
    ids[:len(result)] = result;mask[:len(result)] = 1
    return ids, mask


class TextFusion(nn.Module):
    def __init__(self, snapshot, numeric_width, spec):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(snapshot, local_files_only=True,
                         attn_implementation='eager', add_pooling_layer=False, torch_dtype=torch.float32)
        assert self.backbone.config.model_type == 'xlm-roberta'
        for p in self.backbone.parameters():p.requires_grad_(False)
        assert len(self.backbone.encoder.layer) >= spec['last_layers']
        for layer in self.backbone.encoder.layer[-spec['last_layers']:]:
            for p in layer.parameters():p.requires_grad_(True)
        width = self.backbone.config.hidden_size
        self.text_norm = nn.LayerNorm(width)
        self.numeric_norm = nn.LayerNorm(numeric_width)
        self.head = nn.Sequential(nn.Linear(width+numeric_width, spec['head_width']), nn.GELU(),
                                 nn.Dropout(spec['dropout']), nn.Linear(spec['head_width'], 3))

    def forward(self, ids, mask, numeric):
        hidden = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state.float()
        attention = mask.unsqueeze(-1).float()
        pooled = (hidden*attention).sum(1)/attention.sum(1).clamp(min=1)
        return self.head(torch.cat([self.text_norm(pooled), self.numeric_norm(numeric)], dim=1))


def batch_inputs(ids, masks, numeric, indexes, device):
    length = int(masks[indexes].sum(1).max())
    return (torch.from_numpy(np.asarray(ids[indexes, :length], dtype=np.int64)).to(device),
            torch.from_numpy(np.asarray(masks[indexes, :length], dtype=np.int64)).to(device),
            torch.from_numpy(numeric[indexes]).to(device))


def fit_encoder(ids, masks, numeric, labels, snapshot, spec, folder, work, fp, device, seed, deadline, utils, until_step=None):
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    if device == 'cuda':torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = TextFusion(snapshot, numeric.shape[1], spec).to(device)
    backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    head = [p for n, p in model.named_parameters() if not n.startswith('backbone.')]
    optimizer = torch.optim.AdamW([{'params': backbone, 'lr': spec['encoder_lr']}, {'params': head, 'lr': spec['head_lr']}], weight_decay=spec['weight_decay'])
    steps_per_epoch = math.ceil(len(labels)/spec['batch_size']);total = steps_per_epoch*spec['epochs']
    warmup = max(1, int(total*spec['warmup_fraction']))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step+1)/warmup, max(0, (total-step)/max(1, total-warmup))))
    prep = utils.normalizer(numeric)
    epoch, cursor, step, order, prior_elapsed = 0, 0, 0, None, 0.0
    latest = folder/'latest.json'
    if latest.exists():
        assert verified_marker(latest, fp)
        entry = json.loads(latest.read_text());state = torch.load(folder/entry['checkpoint'], map_location='cpu', weights_only=False)
        assert state['fingerprint'] == fp
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        prep = state['preprocess'];epoch, cursor, step, order = (state[k] for k in ('epoch', 'cursor', 'step', 'order'))
        prior_elapsed = state['elapsed_seconds'];rng.bit_generator.state = state['shuffle_rng']
        torch.set_rng_state(state['torch_rng'])
        if device == 'cuda':torch.cuda.set_rng_state(state['cuda_rng'], device=0)
    x = utils.normalize(numeric, prep);y = np.asarray(labels, dtype=np.int64)
    class_weights = torch.tensor(base.class_weights(y, spec['class_power']), dtype=torch.float32, device=device)
    started = time.monotonic();initial_step = step;last_save = started
    model.train()
    while epoch < spec['epochs']:
        if order is None:order = rng.permutation(len(y));cursor = 0
        indexes = order[cursor:cursor+spec['batch_size']]
        stop_check(work)
        if time.time() >= deadline:raise base.BudgetReached('Hard deadline; latest verified optimizer retained')
        inputs = batch_inputs(ids, masks, x, indexes, device)
        targets = torch.from_numpy(y[indexes]).to(device)
        optimizer.zero_grad(set_to_none=True)
        precision = torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' and spec['precision'] == 'bfloat16' else nullcontext()
        with precision:
            loss = nn.functional.cross_entropy(model(*inputs).float(), targets, weight=class_weights)
        if not torch.isfinite(loss):raise ValueError('Non-finite encoder loss')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(), spec['gradient_clip']);optimizer.step();scheduler.step()
        cursor += len(indexes);step += 1
        end_epoch = cursor >= len(y)
        if end_epoch:epoch += 1;cursor = 0;order = None
        elapsed = time.monotonic()-started
        if step % 10 == 0 or end_epoch:
            atomic_json(work/'neural-progress.json', {'candidate': spec['name'], 'job': folder.name, 'epoch': epoch,
                'iterations': step, 'total_iterations': total, 'session_iterations': step-initial_step,
                'session_elapsed_seconds': elapsed, 'elapsed_seconds': prior_elapsed+elapsed, 'at': now()})
        test_pause = until_step is not None and step >= until_step
        if step % spec['checkpoint_steps'] == 0 or time.monotonic()-last_save >= 60 or end_epoch or test_pause:
            state = {'fingerprint': fp, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'preprocess': prep, 'epoch': epoch, 'cursor': cursor,
                     'step': step, 'order': order, 'shuffle_rng': rng.bit_generator.state,
                     'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(0) if device == 'cuda' else None,
                     'elapsed_seconds': prior_elapsed+time.monotonic()-started}
            path = folder/f'optimizer-step-{step:07d}.pt';utils.atomic_torch(path, state)
            mark(latest, fp, [path], checkpoint=path.name, step=step, epoch=epoch)
            if end_epoch:
                epoch_path = folder/f'epoch-end-{epoch:03d}.pt'
                if not epoch_path.exists():os.link(path, epoch_path)
            # Keep the latest and previous atomic optimizer plus immutable epoch-end links.
            for old in sorted(folder.glob('optimizer-step-*.pt'))[:-2]:old.unlink()
            last_save = time.monotonic()
        if test_pause:break
    model.zero_grad(set_to_none=True);model.eval()
    return model, prep


def predict_encoder(model, prep, ids, masks, numeric, device, spec, utils):
    numeric = utils.normalize(numeric, prep);values = []
    with torch.inference_mode():
        for offset in range(0, len(ids), spec['batch_size']*2):
            indexes = np.arange(offset, min(offset+spec['batch_size']*2, len(ids)))
            inputs = batch_inputs(ids, masks, numeric, indexes, device)
            precision = torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' and spec['precision'] == 'bfloat16' else nullcontext()
            with precision:logits = model(*inputs)
            values.append(torch.softmax(logits.float(), 1).cpu().numpy())
    return np.concatenate(values)


def build_inputs(work, snapshot, config, plan, experiment, fp):
    p = work/'cache/prepared.json';state = json.loads(p.read_text());assert verified_marker(p, state['fingerprint'])
    train, test = (pd.read_parquet(work/f'cache/{name}.parquet') for name in ('train', 'test'))
    assert len(train) == config['expected']['train'] and len(test) == config['expected']['test']
    assert 'label' not in test and not set(train.author) & set(test.author) and train.groupby('author').fold.nunique().max() == 1
    extra_path = work/plan['past_features'];assert extra_path.resolve().is_relative_to(work.resolve()) and digest(extra_path) == plan['past_features_sha256']
    extra = pd.read_parquet(extra_path);names = [c for c in extra if c not in KEYS]
    assert all(c.startswith(('rh_', 'sr_')) for c in names)
    columns = json.loads((work/'cache/feature_columns.json').read_text())['numeric']
    frames = [df.merge(extra, on=KEYS, validate='one_to_one') for df in (train, test)]
    posts = pd.read_parquet(work/'cache/all_posts.parquet', columns=KEYS);index = pd.MultiIndex.from_frame(posts)
    image_fp = identity({'settings': config['image_encoder'], 'data': digest(work/'cache/all_posts.parquet')})
    assert verified_marker(work/'cache/image_embeddings.json', image_fp)
    images = np.load(work/'cache/image_embeddings.npy', allow_pickle=False, mmap_mode='r')
    numeric = []
    for frame in frames:
        indexes = index.get_indexer(pd.MultiIndex.from_frame(frame[KEYS]));assert (indexes >= 0).all()
        values = frame[columns+names].to_numpy(dtype=np.float32)
        numeric.append(np.concatenate([values, np.isnan(values).astype(np.float32), images[indexes]], axis=1))
    # All candidates in this bounded group share one fixed token policy/length.
    lengths = {s['max_length'] for s in plan['candidates']};assert len(lengths) == 1;length = lengths.pop()
    token_marker = experiment/'tokens.json';token_fp = identity({'parent': fp, 'max_length': length, 'policy': 'body-document-head-tail'})
    if not verified_marker(token_marker, token_fp):
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        assert tokenizer.cls_token_id is not None and tokenizer.sep_token_id is not None and tokenizer.pad_token_id is not None
        arrays, truncated = [], 0
        for frame in frames:
            texts = (frame.text.fillna('').map(canonical_text)+' document '+frame.document_text.fillna('').str.slice(0,2000).map(canonical_text)).tolist()
            ids, masks = np.zeros((len(frame), length), np.int32), np.zeros((len(frame), length), np.int8)
            for offset in range(0, len(texts), 256):
                stop_check(work)
                tokens = tokenizer(texts[offset:offset+256], add_special_tokens=False, truncation=False,
                                   return_attention_mask=False, return_token_type_ids=False, verbose=False)['input_ids']
                for j, raw in enumerate(tokens):
                    ids[offset+j], masks[offset+j] = pack_tokens(raw, tokenizer, length);truncated += len(raw)>length-2
            arrays += [ids, masks]
        paths = [experiment/f'{name}.npy' for name in ('train-ids','train-mask','test-ids','test-mask')]
        for path, value in zip(paths, arrays):save_array(path, value)
        mark(token_marker, token_fp, paths, train_rows=len(train), test_rows=len(test), max_length=length, head_tail_truncated=truncated)
    arrays = [np.load(experiment/f'{name}.npy', allow_pickle=False, mmap_mode='r') for name in ('train-ids','train-mask','test-ids','test-mask')]
    return frames[0], frames[1], (arrays[0], arrays[1], numeric[0]), (arrays[2], arrays[3], numeric[1])


def evaluate(spec, config, train, test, xtrain, xtest, snapshot, work, fp_group, device, budget_end, utils):
    folder = work/'models'/spec['name'];folder.mkdir(parents=True, exist_ok=True);fp = identity({'parent': fp_group, 'spec': spec})
    if verified_marker(folder/'complete.json', fp):return np.load(folder/'probabilities.npz', allow_pickle=False)
    hard_end = datetime.fromisoformat(config['deadline']).timestamp() if config.get('deadline') else float('inf')
    oof = np.zeros((len(train), 3), np.float32)
    for fold in sorted(train.fold.unique()):
        valid = np.flatnonzero(train.fold.to_numpy() == fold);fit = np.flatnonzero(train.fold.to_numpy() != fold)
        fold_fp = identity({'parent': fp, 'fold': int(fold)});marker = folder/f'fold-{fold}.json';probs = folder/f'fold-{fold}.npz'
        if not verified_marker(marker, fold_fp):
            stop_check(work)
            if time.time() >= budget_end:raise base.BudgetReached('Original search budget reached')
            assert verified_marker(snapshot/'verified.json', identity(config['text_encoder']))
            atomic_json(work/'training-status.json', {'candidate': spec['name'], 'job': f'fold-{fold}', 'stage': 'fit', 'optimizer': 'pytorch', 'at': now()})
            started = time.monotonic()
            model, prep = fit_encoder(*(a[fit] for a in xtrain), train.iloc[fit].label, snapshot, spec,
                         folder/'checkpoints'/f'fold-{fold}', work, fold_fp, device, config['seed'], hard_end, utils)
            values = predict_encoder(model, prep, *(a[valid] for a in xtrain), device, spec, utils);base.check_probabilities(values, len(valid))
            utils.atomic_torch(folder/f'fold-{fold}.pt', {'weights': model.state_dict(), 'preprocess': prep, 'spec': spec})
            with probs.with_suffix('.tmp').open('wb') as stream:np.savez_compressed(stream, probabilities=values)
            probs.with_suffix('.tmp').replace(probs);mark(marker, fold_fp, [probs, folder/f'fold-{fold}.pt'], elapsed_seconds=time.monotonic()-started)
            del model;gc.collect()
            if device == 'cuda':torch.cuda.empty_cache()
        with np.load(probs, allow_pickle=False) as saved:oof[valid] = saved['probabilities']
    stop_check(work)
    assert verified_marker(snapshot/'verified.json', identity(config['text_encoder']))
    atomic_json(work/'training-status.json', {'candidate': spec['name'], 'job': 'full', 'stage': 'full_fit', 'optimizer': 'pytorch', 'at': now()})
    model, prep = fit_encoder(*xtrain, train.label, snapshot, spec, folder/'checkpoints/full', work, fp, device, config['seed'], hard_end, utils)
    values = predict_encoder(model, prep, *xtest, device, spec, utils);base.check_probabilities(values, len(test))
    utils.atomic_torch(folder/'model.pt', {'weights': model.state_dict(), 'preprocess': prep, 'spec': spec})
    p = folder/'probabilities.npz'
    with p.with_suffix('.tmp').open('wb') as stream:np.savez_compressed(stream, oof=oof, test=values)
    p.with_suffix('.tmp').replace(p)
    atomic_json(folder/'metrics.json', {'selection_split': 'train_author_oof', 'spec': spec, 'oof': base.score(train.label, oof.argmax(1)),
                'folds': [base.score(train.loc[train.fold == fold,'label'], oof[train.fold == fold].argmax(1)) for fold in sorted(train.fold.unique())]})
    mark(folder/'complete.json', fp, [folder/'model.pt', p, folder/'metrics.json'])
    del model;gc.collect()
    if device == 'cuda':torch.cuda.empty_cache()
    return np.load(p, allow_pickle=False)


def run(work, models, plan_path, group, helper_path, utils_path, device='cuda', gpu_uuid=None):
    stop_check(work);assert not (work/'report/official-test.json').exists()
    parent = json.loads((work/'environment/manifest.json').read_text());config = parent['config']
    assert parent['fingerprint'] == identity({'code': artifacts_fingerprint(Path(base.__file__).resolve().parents[1]), 'config': config})
    plan = json.loads(plan_path.read_text());assert 1 <= len(plan['candidates']) <= 2
    assert re.fullmatch(r'[a-z0-9-]{1,32}', group)
    assert len({s['name'] for s in plan['candidates']}) == len(plan['candidates'])
    assert all(s['name'].startswith(f'refine-{group}-') and s['family'] == 'finetune' for s in plan['candidates'])
    snapshot = model_path(models, 'text_encoder', config['text_encoder'])
    torch.set_num_threads(config['threads']);torch.use_deterministic_algorithms(True)
    if device == 'cuda':
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
        torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
        assert torch.ones(1024, device='cuda').sum().item() == 1024
    helper, utils = module(helper_path, 'ft_blend_helper'), module(utils_path, 'ft_nn_utils')
    inputs = {name: digest(work/'cache'/name) for name in ('train.parquet','test.parquet','prepared.json','image_embeddings.json')}
    fp = identity({'parent': parent['fingerprint'], 'inputs': inputs, 'plan': plan, 'script': digest(Path(__file__)),
                   'helper': digest(helper_path), 'utils': digest(utils_path), 'snapshot_marker': digest(snapshot/'verified.json'), 'device_type': device})
    experiment = work/'environment/refinements'/group;experiment.mkdir(parents=True, exist_ok=True);marker = experiment/'manifest.json'
    if marker.exists():assert json.loads(marker.read_text())['fingerprint'] == fp
    else:
        atomic_json(marker, {'at': now(), 'fingerprint': fp, 'scientific_extension': True, 'parent_fingerprint': parent['fingerprint'],
                    'inputs': inputs, 'plan': plan, 'script_sha256': digest(Path(__file__)), 'helper_sha256': digest(helper_path),
                    'utils_sha256': digest(utils_path), 'snapshot_marker_sha256': digest(snapshot/'verified.json'),
                    'test_targets_used': False, 'author_folds_preserved': True, 'same_search_budget': json.loads((work/'search-budget.json').read_text())})
        shutil.copy2(work/'selection.json', experiment/'selection-before.json');shutil.copy2(work/'predictions/final.csv', experiment/'predictions-before.csv')
    with (experiment/'runtime-history.jsonl').open('a') as stream:stream.write(json.dumps({'at': now(), 'device': device, 'gpu_uuid_from_launcher': gpu_uuid,
                     'gpu_name': torch.cuda.get_device_name(0) if device == 'cuda' else None, 'torch': str(torch.__version__), 'cuda_check': 'passed' if device == 'cuda' else 'not-requested'})+'\n')
    train, test, xtrain, xtest = build_inputs(work, snapshot, config, plan, experiment, fp)
    predictions = helper.load_predictions(work);budget_end = json.loads((work/'search-budget.json').read_text())['ends'];trials = []
    for spec in plan['candidates']:
        stop_check(work);assert not (work/'report/official-test.json').exists()
        if time.time() >= budget_end:break
        try:
            with evaluate(spec, config, train, test, xtrain, xtest, snapshot, work, fp, device, budget_end, utils) as values:
                predictions[spec['name']] = {k: values[k].copy() for k in ('oof','test')}
        except base.BudgetReached:break
        helper.publish_best(work, train, test, predictions)
        trials.append({'candidate': spec['name'], 'metrics': json.loads((work/'models'/spec['name']/'metrics.json').read_text())['oof'],
                       'selected_metrics': json.loads((work/'selection.json').read_text())['metrics']})
        atomic_json(experiment/'progress.json', {'at': now(), 'trials': trials, 'best_selection': json.loads((work/'selection.json').read_text())})
        if trials[-1]['selected_metrics']['macro_f1'] >= .70:break
    oof, _, scales = helper.publish_best(work, train, test, predictions);selected = json.loads((work/'selection.json').read_text())
    report = {'selection': selected, 'candidates': {n: json.loads((work/'models'/n/'metrics.json').read_text()) for n in predictions},
              'confusion_train_oof': confusion_matrix(train.label, (oof*scales).argmax(1), labels=[0,1,2]).tolist(),
              'per_class_train_oof': classification_report(train.label, (oof*scales).argmax(1), output_dict=True, zero_division=0),
              'scientific_extensions': [str(p.relative_to(work)) for p in sorted((work/'environment/refinements').glob('*/manifest.json'))]}
    report.update(base.diagnostics(train, (oof*scales).argmax(1), predictions['baseline']['oof'].argmax(1), config['bootstrap_samples'], config['seed']))
    atomic_json(work/'report/training.json', report)
    lines = ['# Kairos models and supervised text adaptation', '', 'Model selection uses train-author OOF; official test remains unseen.', '', '| Candidate | OOF macro-F1 | QWK |', '| --- | --- | --- |']
    for name, result in report['candidates'].items():lines.append(f"| {name} | {result['oof']['macro_f1']:.4f} | {result['oof']['qwk']:.4f} |")
    lines += ['', 'Text adaptation loads the pinned local multilingual encoder; only the declared last layers and numeric fusion head are trained.',
              'Current body/document tokens use a bounded head/tail policy; numerical preprocessing fits only fold training rows.',
              'Optimizer/scheduler/shuffle/dropout state is saved at the declared step interval, approximately60seconds apart and at epoch boundaries.',
              'OOF is reused for selection and can be optimistic; collection-time features are retrospective.']
    (work/'report/REPORT.md').write_text('\n'.join(lines)+'\n')
    atomic_json(experiment/'complete.json', {'at': now(), 'fingerprint': fp, 'trials': trials, 'best_selection': selected})
    return selected


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--work', type=Path, default=Path('/work'));parser.add_argument('--models', type=Path, default=Path('/models'))
    parser.add_argument('--plan', type=Path, required=True);parser.add_argument('--group', required=True);parser.add_argument('--helper', type=Path, required=True)
    parser.add_argument('--nn-utils', type=Path, required=True);parser.add_argument('--gpu-uuid', required=True);args = parser.parse_args()
    atomic_json(args.work/'status.json', {'state': 'running', 'stage': 'train', 'extension': args.group, 'at': now()})
    try:
        result = run(args.work, args.models, args.plan, args.group, args.helper, args.nn_utils, gpu_uuid=args.gpu_uuid)
        atomic_json(args.work/'status.json', {'state': 'completed', 'stage': 'refinement', 'extension': args.group, 'at': now(), 'selection': result})
    except Exception as exc:
        atomic_json(args.work/'status.json', {'state': 'needs_attention', 'stage': 'refinement', 'extension': args.group, 'at': now(), 'error_type': type(exc).__name__, 'reason': str(exc)})
        raise
