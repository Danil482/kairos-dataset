#!/usr/bin/env python3
"""Raw mention identities and strictly earlier graph context, with author OOF."""
import argparse
import importlib.util
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

import kairos_lab.train as base
from kairos_lab.common import artifacts_fingerprint, atomic_json, digest, identity, now, stop_check, verified_marker
from kairos_lab.dataset import KEYS
from kairos.label import fit_cutoffs

ENTITIES = re.compile(r'⟨(ORG|AUTHOR|PERSON)_(\d+)⟩')


def load_raw_posts(data, work=None):
    """Allowlist current text/keys only; never copy raw current outcome fields."""
    rows, sources = [], {}
    for path in sorted((data / 'posts').glob('AUTHOR_*.json')):
        if work is not None:
            stop_check(work)
        sources[str(path.relative_to(data))] = digest(path)
        feed = json.loads(path.read_text());owner = feed['author']
        for post in feed['posts']:
            if isinstance(post.get('author'), dict) and post['author'].get('code') == owner:
                rows.append({'author': owner, 'post': post['urn'], 'raw_text': post.get('text') or ''})
    result = pd.DataFrame(rows, columns=KEYS + ['raw_text'])
    assert len(result) and not result.duplicated(KEYS).any()
    return result, identity(sources)


def entities(text, author):
    # PERSON numbering is local to a feed; ORG/AUTHOR identities are global.
    return {(kind, str(int(number)), author if kind == 'PERSON' else '')
            for kind, number in ENTITIES.findall(str(text))}


def native_raw_text(values, helper):
    # Keep global codes, but never pretend equal PERSON IDs link different feeds.
    values = values.fillna('').str.replace(r'⟨PERSON_\d+⟩', '⟨PERSON⟩', regex=True)
    return helper.native_text(values, True)


def graph_features(posts, outcomes, raw, cutoffs, work=None):
    """Global structure uses no outcomes; matched outcome histories are own-author only."""
    assert posts[KEYS].equals(outcomes[KEYS])
    frame = posts[KEYS + ['posted_at', 'level', 'history_count']].merge(
        raw, on=KEYS, how='left', sort=False, validate='one_to_one')
    assert len(frame) == len(raw) and frame.raw_text.notna().all()
    times = frame.posted_at.astype('int64').to_numpy()
    codes = [entities(text, author) for text, author in zip(frame.raw_text, frame.author)]
    logs = np.log1p(np.maximum(outcomes.reactions.to_numpy(float), 0))
    residuals = logs - frame.level.to_numpy(float)
    residuals[frame.history_count.to_numpy() < 5] = np.nan
    order = np.argsort(times, kind='stable');features = [None] * len(frame)
    own_edges = defaultdict(list);own_nodes = defaultdict(set);own_post_count = defaultdict(int)
    global_posts = defaultdict(int);global_authors = defaultdict(set);global_last = {}
    prior = np.array([.2, .6, .2]);low, high = cutoffs
    start = 0
    while start < len(order):
        if work is not None:
            stop_check(work)
        end = start + 1
        while end < len(order) and times[order[end]] == times[order[start]]:
            end += 1
        batch = order[start:end]
        # Read the complete timestamp batch before inserting any of its edges/outcomes.
        for i in batch:
            author = frame.iloc[i].author;current = codes[i]
            row = {'eg_current_entities': len(current), 'eg_author_prior_nodes': len(own_nodes[author]),
                   'eg_author_prior_posts': own_post_count[author]}
            for kind in ('ORG', 'AUTHOR', 'PERSON'):
                row['eg_current_' + kind.lower()] = sum(k[0] == kind for k in current)
            seen = [own_edges[(author, key)] for key in sorted(current)]
            row['eg_own_new_fraction'] = np.mean([not x for x in seen]) if seen else 0.
            counts = [len(x) for x in seen]
            row['eg_own_edge_mean'] = np.mean(counts) if counts else 0.
            row['eg_own_edge_max'] = max(counts, default=0)
            matched = sorted({j for indexes in seen for j in indexes}, key=lambda j: (times[j], j))
            row['eg_matched_prior_posts'] = len(matched)
            lags = [(times[i] - times[x[-1]]) / 86400e9 for x in seen if x]
            row['eg_own_last_gap_min'] = min(lags, default=np.nan)
            row['eg_own_last_gap_mean'] = np.mean(lags) if lags else np.nan
            eligible = [j for j in matched if np.isfinite(residuals[j])]
            for window in (5, 20):
                values = residuals[eligible[-window:]]
                row[f'eg_residual_mean{window}'] = values.mean() if len(values) else np.nan
                row[f'eg_residual_std{window}'] = values.std() if len(values) else np.nan
                classes = np.where(values < low, 0, np.where(values > high, 2, 1))
                totals = np.bincount(classes, minlength=3)
                for k in range(3):
                    row[f'eg_class{window}_{k}'] = (totals[k] + 2 * prior[k]) / (len(values) + 2)
            row['eg_matched_centered_log5'] = (logs[matched[-5:]].mean() - frame.iloc[i].level
                                              if matched else np.nan)
            global_keys = [k for k in current if k[0] != 'PERSON']
            frequencies = [global_posts[k] for k in global_keys]
            degrees = [len(global_authors[k]) for k in global_keys]
            row['eg_global_new_fraction'] = np.mean([n == 0 for n in frequencies]) if frequencies else 0.
            row['eg_global_mentions_mean'] = np.mean(frequencies) if frequencies else 0.
            row['eg_global_mentions_max'] = max(frequencies, default=0)
            row['eg_global_authors_mean'] = np.mean(degrees) if degrees else 0.
            row['eg_global_authors_max'] = max(degrees, default=0)
            gaps = [(times[i] - global_last[k]) / 86400e9 for k in global_keys if k in global_last]
            row['eg_global_last_gap_min'] = min(gaps, default=np.nan)
            features[i] = row
        for i in batch:
            author = frame.iloc[i].author;own_post_count[author] += 1
            for key in codes[i]:
                own_edges[(author, key)].append(i);own_nodes[author].add(key)
                if key[0] != 'PERSON':
                    global_posts[key] += 1;global_authors[key].add(author);global_last[key] = times[i]
        start = end
    result = pd.DataFrame(features);result[KEYS] = posts[KEYS]
    return result


def run(work, data, plan_path, group, helper_path):
    stop_check(work);assert not (work / 'report/official-test.json').exists()
    parent = json.loads((work / 'environment/manifest.json').read_text());config = parent['config']
    assert parent['fingerprint'] == identity({'code': artifacts_fingerprint(Path(base.__file__).resolve().parents[1]), 'config': config})
    spec = importlib.util.spec_from_file_location('entity_native_helper', helper_path)
    helper = importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    plan = json.loads(plan_path.read_text());plans = plan['candidates']
    assert re.fullmatch(r'r\d\d', group) and 1 <= len(plans) <= 3
    assert len({s['name'] for s in plans}) == len(plans)
    assert all(s['name'].startswith(f'refine-{group}-') and s['family'] == 'multimodal' for s in plans)
    raw, raw_sha = load_raw_posts(data, work)
    raw_entity_posts = int(raw.raw_text.str.contains(r'⟨(?:ORG|AUTHOR|PERSON)_\d+⟩', regex=True).sum())
    assert raw_entity_posts > 0, 'Original raw entity IDs are required for this separate experiment'
    inputs = {n: digest(work / 'cache' / n) for n in ('train.parquet', 'test.parquet', 'all_posts.parquet', 'history_outcomes.parquet')}
    fp = identity({'parent': parent['fingerprint'], 'inputs': inputs, 'raw_posts': raw_sha,
                   'script': digest(Path(__file__)), 'helper': digest(helper_path), 'plan': plan})
    e = work / 'environment/refinements' / group;e.mkdir(parents=True, exist_ok=True)
    if (e / 'manifest.json').exists():
        assert json.loads((e / 'manifest.json').read_text())['fingerprint'] == fp
    else:
        atomic_json(e / 'manifest.json', {'at': now(), 'fingerprint': fp, 'parent_fingerprint': parent['fingerprint'],
                    'inputs': inputs, 'raw_posts_fingerprint': raw_sha, 'script_sha256': digest(Path(__file__)),
                    'helper_sha256': digest(helper_path), 'plan': plan, 'scientific_extension': True,
                    'test_targets_used': False, 'author_folds_preserved': True,
                    'global_graph': 'strictly earlier unlabeled own-post mention structure; no cross-author outcomes',
                    'own_entity_history': 'strictly earlier same-author outcomes only; tied timestamps excluded',
                    'same_search_budget': json.loads((work / 'search-budget.json').read_text())})
    train, test, numeric, extra, embeddings, past_sha = helper.build_frames(work, config, e)
    assert past_sha == plan['past_features_sha256']
    posts = pd.read_parquet(work / 'cache/all_posts.parquet');outcomes = pd.read_parquet(work / 'cache/history_outcomes.parquet')
    assert len(raw) == len(posts) and not raw.duplicated(KEYS).any()
    mask = posts.author.isin(set(train.author)) & posts.history_count.ge(5)
    cutoffs = fit_cutoffs(pd.Series(np.log1p(np.maximum(outcomes.reactions.to_numpy(float), 0)) - posts.level)[mask])
    graph = graph_features(posts, outcomes, raw, cutoffs, work)
    path = e / 'entity-graph-features.parquet';graph.to_parquet(path, index=False);graph_sha = digest(path)
    graph_columns = [c for c in graph if c not in KEYS]
    train = train.merge(graph, on=KEYS, validate='one_to_one').merge(raw, on=KEYS, validate='one_to_one')
    test = test.merge(graph, on=KEYS, validate='one_to_one').merge(raw, on=KEYS, validate='one_to_one')
    assert len(train) == config['expected']['train'] and len(test) == config['expected']['test'] and 'label' not in test
    atomic_json(e / 'feature-audit.json', {'at': now(), 'raw_rows': len(raw), 'raw_entity_posts': raw_entity_posts,
                'columns': graph_columns, 'feature_sha256': graph_sha, 'past_feature_sha256': past_sha,
                'cutoffs': list(cutoffs), 'raw_ids_from_original_inputs': True, 'future_or_tied_graph_edges_used': False})
    predictions = helper.load_predictions(work);results = [];budget_end = json.loads((work / 'search-budget.json').read_text())['ends']
    original = base.estimator;base.estimator = helper.extended_estimator
    try:
        for requested in plans:
            stop_check(work);assert not (work / 'report/official-test.json').exists()
            if time.time() >= budget_end:
                break
            trial = dict(requested);trial.update(extension_fingerprint=fp, feature_sha256=identity([past_sha, graph_sha]))
            for frame in (train, test):
                if trial.get('native_text'):
                    frame['native_text'] = native_raw_text(frame.raw_text + ' profile ' + frame.profile_text.fillna('').str.slice(0, 1500), helper)
            selected = embeddings
            if trial.get('embedding_mode') == 'current-text':
                width = (sum(c.startswith('text_encoder_component_') for c in embeddings) - 8) // 2
                selected = [f'text_encoder_component_{i}' for i in range(width)]
            try:
                with base.evaluate_candidate(trial, config, train, test, numeric + extra + graph_columns, work, budget_end, selected) as values:
                    predictions[trial['name']] = {k: values[k].copy() for k in ('oof', 'test')}
            except base.BudgetReached:
                break
            helper.publish_best(work, train, test, predictions)
            results.append({'candidate': trial['name'], 'metrics': json.loads((work / 'models' / trial['name'] / 'metrics.json').read_text())['oof'],
                            'selected_metrics': json.loads((work / 'selection.json').read_text())['metrics']})
            atomic_json(e / 'progress.json', {'at': now(), 'trials': results})
            if results[-1]['selected_metrics']['macro_f1'] >= .70:
                break
    finally:
        base.estimator = original
    oof, _, scales = helper.publish_best(work, train, test, predictions)
    selected = json.loads((work / 'selection.json').read_text())
    report = {'selection': selected, 'candidates': {n: json.loads((work / 'models' / n / 'metrics.json').read_text()) for n in predictions},
              'confusion_train_oof': confusion_matrix(train.label, (oof * scales).argmax(1), labels=[0, 1, 2]).tolist(),
              'per_class_train_oof': classification_report(train.label, (oof * scales).argmax(1), output_dict=True, zero_division=0),
              'scientific_extensions': [str(p.relative_to(work)) for p in sorted((work / 'environment/refinements').glob('*/manifest.json'))]}
    report.update(base.diagnostics(train, (oof * scales).argmax(1), predictions['baseline']['oof'].argmax(1), config['bootstrap_samples'], config['seed']))
    atomic_json(work / 'report/training.json', report)
    lines = ['# Kairos: raw entity and temporal graph refinement', '', 'Model selection uses train-author OOF; official test remains unseen.', '',
             '| Candidate | OOF macro-F1 | QWK |', '| --- | --- | --- |']
    for name, result in report['candidates'].items():
        lines.append(f"| {name} | {result['oof']['macro_f1']:.4f} | {result['oof']['qwk']:.4f} |")
    lines += ['', 'Raw ORG/AUTHOR mention identities come from original inputs; PERSON identifiers are local to each feed.',
              'All graph edges precede the row timestamp; tied-date batches update afterwards. Global mention structure uses no outcomes.',
              'Entity-conditioned outcome histories use the same author only. No current outcomes or test targets enter modeling.',
              'OOF is reused for selection and can be optimistic; earlier prepared-text entity variants could not recover raw IDs.']
    (work / 'report/REPORT.md').write_text('\n'.join(lines) + '\n')
    atomic_json(e / 'complete.json', {'at': now(), 'fingerprint': fp, 'trials': results, 'best_selection': selected})
    return selected


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=Path('/work'));parser.add_argument('--data', type=Path, default=Path('/inputs'))
    parser.add_argument('--plan', type=Path, required=True);parser.add_argument('--group', required=True);parser.add_argument('--helper', type=Path, required=True)
    args = parser.parse_args();atomic_json(args.work / 'status.json', {'state': 'running', 'stage': 'train', 'extension': args.group, 'at': now()})
    try:
        result = run(args.work, args.data, args.plan, args.group, args.helper)
        atomic_json(args.work / 'status.json', {'state': 'completed', 'stage': 'refinement', 'extension': args.group, 'at': now(), 'selection': result})
    except Exception as exc:
        atomic_json(args.work / 'status.json', {'state': 'needs_attention', 'stage': 'refinement', 'extension': args.group, 'at': now(), 'error_type': type(exc).__name__, 'reason': str(exc)})
        raise
