#!/usr/bin/env python3
"""Owner-authorized scientific extensions; frozen parent inputs, train OOF only."""
import argparse
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

import kairos_lab.train as base
from kairos_lab.common import artifacts_fingerprint, atomic_json, digest, identity, now, stop_check, verified_marker
from kairos_lab.dataset import KEYS
from kairos_lab.encoders import contextual_embeddings

ORIGINAL_ESTIMATOR = base.estimator


def rich_history(posts, outcomes, cutoffs, work=None):
    """Lagged residual/streak/transition features; current outcomes never enter."""
    assert posts[KEYS].equals(outcomes[KEYS])
    low, high = cutoffs
    prior = np.array([.2, .6, .2])
    rows = [None] * len(posts)
    for _, group in posts.groupby('author', sort=False):
        if work is not None:
            stop_check(work)
        idx = group.index.to_numpy()
        times = group.posted_at.astype('int64').to_numpy()
        logs = np.log1p(np.maximum(outcomes.iloc[idx].reactions.to_numpy(float), 0))
        residuals = logs - group.level.to_numpy(float)
        residuals[group.history_count.to_numpy() < 5] = np.nan
        for j, current in enumerate(idx):
            end = int(np.searchsorted(times, times[j], side='left'))
            past = residuals[:end]
            past = past[np.isfinite(past)]
            row = {f'rh_{name}': np.nan for name in ('last_log_delta', 'last_residual', 'residual_mean5',
                   'residual_mean20', 'residual_std20', 'residual_ewm3', 'residual_ewm10', 'streak')}
            if end:
                row['rh_last_log_delta'] = logs[end-1] - float(group.iloc[j].level)
            classes = np.where(past < low, 0, np.where(past > high, 2, 1))
            for window in (5, 20, 0):
                values = classes[-window:] if window else classes
                counts = np.bincount(values, minlength=3)
                for k in range(3):
                    row[f'rh_class_{window}_{k}'] = (counts[k] + 2 * prior[k]) / (len(values) + 2)
            transition = np.zeros(3)
            if len(classes) > 1:
                transition = np.bincount(classes[1:][classes[:-1] == classes[-1]], minlength=3)
            for k in range(3):
                row[f'rh_transition_{k}'] = (transition[k] + 3 * prior[k]) / (transition.sum() + 3)
            if len(past):
                row.update(rh_last_residual=float(past[-1]), rh_residual_mean5=float(past[-5:].mean()),
                           rh_residual_mean20=float(past[-20:].mean()), rh_residual_std20=float(past[-20:].std()))
                recent = past[-50:]
                ages = np.arange(len(recent)-1, -1, -1)
                for span in (3, 10):
                    row[f'rh_residual_ewm{span}'] = float(np.average(recent, weights=np.exp2(-ages/span)))
                streak = 0
                for value in classes[::-1]:
                    if value != classes[-1]: break
                    streak += 1
                row['rh_streak'] = streak
            if end:
                centered = logs[:end] - float(group.iloc[j].level)
                historical = np.where(centered < low, 0, np.where(centered > high, 2, 1))
                counts = np.bincount(historical, minlength=3)
            else:
                counts = np.zeros(3)
            for k in range(3):
                row[f'rh_centered_class_{k}'] = (counts[k] + 2 * prior[k]) / (end + 2)
            rows[current] = row
    result = pd.DataFrame(rows)
    result[KEYS] = posts[KEYS]
    return result


def retrieval_history(posts, outcomes, raw_text, raw_images, cutoffs, work=None):
    """Semantic neighbors from this author's strictly earlier posts only (last 300)."""
    assert posts[KEYS].equals(outcomes[KEYS])
    assert len(posts) == len(raw_text) == len(raw_images)
    text = np.asarray(raw_text, dtype=np.float32)
    text = text / np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-12)
    images = np.asarray(raw_images[:, :-1], dtype=np.float32)
    images = images / np.maximum(np.linalg.norm(images, axis=1, keepdims=True), 1e-12)
    rows = [None] * len(posts)
    low, high = cutoffs
    for _, group in posts.groupby('author', sort=False):
        if work is not None:
            stop_check(work)
        idx = group.index.to_numpy()
        times = group.posted_at.astype('int64').to_numpy()
        assert (np.diff(times) >= 0).all()
        logs = np.log1p(np.maximum(outcomes.iloc[idx].reactions.to_numpy(float), 0))
        levels = group.level.to_numpy(float)
        residuals = logs - levels
        residuals[group.history_count.to_numpy() < 5] = np.nan
        for j, current in enumerate(idx):
            end = int(np.searchsorted(times, times[j], side='left'))
            begin = max(0, end-300)
            prior_idx = idx[begin:end]
            row = {}
            for modality in ('text', 'image'):
                for k in (3, 10, 30):
                    for name in ('sim_max','sim_mean','centered_mean','centered_std','prior_residual','age_days','class0','class1','class2'):
                        row[f'sr_{modality}_{k}_{name}'] = np.nan
            if end:
                ages = (times[j] - times[begin:end]) / 86400e9
                centered = logs[begin:end] - levels[j]
                for modality, embeddings in (('text', text), ('image', images)):
                    valid = np.isfinite(centered)
                    if modality == 'image':
                        valid &= raw_images[prior_idx, -1] > 0
                        if raw_images[current, -1] <= 0:
                            continue
                    if not valid.any():
                        continue
                    similarities = embeddings[prior_idx] @ embeddings[current]
                    order = np.flatnonzero(valid)[np.argsort(similarities[valid], kind='stable')[::-1]]
                    for k in (3,10,30):
                        selected = order[:k]
                        sims = similarities[selected]
                        weights = np.exp(8*(sims-sims.max()))
                        values = centered[selected]
                        mean = float(np.average(values, weights=weights))
                        prefix = f'sr_{modality}_{k}_'
                        row.update({prefix+'sim_max':float(sims.max()),prefix+'sim_mean':float(sims.mean()),
                                    prefix+'centered_mean':mean,prefix+'centered_std':float(np.sqrt(np.average((values-mean)**2,weights=weights))),
                                    prefix+'age_days':float(np.average(ages[selected], weights=weights))})
                        prior = residuals[begin:end][selected]
                        eligible = np.isfinite(prior)
                        if eligible.any():
                            row[prefix+'prior_residual'] = float(np.average(prior[eligible],weights=weights[eligible]))
                        classes = np.where(values < low, 0, np.where(values > high, 2, 1))
                        for c in range(3):
                            row[prefix+f'class{c}'] = float((weights[classes==c].sum()+np.array([.2,.6,.2])[c])/(weights.sum()+1))
            rows[current] = row
    result = pd.DataFrame(rows)
    result[KEYS] = posts[KEYS]
    return result


def native_text(values, retain_entities):
    import re
    def clean(value):
        text = str(value)
        if retain_entities:
            text = re.sub(r'⟨(PERSON|ORG|AUTHOR)_(\d+)⟩', lambda m: ' entity'+m[1].lower()+m[2]+' ', text)
        else:
            from kairos_lab.dataset import canonical_text
            text = canonical_text(text)
        text = re.sub(r'https?://\S+', ' link ', text)
        tokens = re.findall(r'[\w]+', text.lower(), flags=re.UNICODE)
        return ' '.join(tokens) or 'emptytext'
    return values.fillna('').map(clean)


def extended_estimator(spec, config, numeric, y, emb_columns=()):
    model = ORIGINAL_ESTIMATOR(spec, config, numeric, y, emb_columns)
    if emb_columns and 'pca_components' in spec:
        model.set_params(prep__emb__n_components=min(spec['pca_components'], len(emb_columns), len(y)-1))
    if 'l2_leaf_reg' in spec:
        model.set_params(clf__l2_leaf_reg=spec['l2_leaf_reg'])
    if spec.get('native_text'):
        prep = model.named_steps['prep']
        prep.transformers.append(('text', 'passthrough', ['native_text']))
        prep.set_output(transform='pandas')
        model.set_params(clf__text_features=['text__native_text'],
            clf__tokenizers=[{'tokenizer_id':'Space','separator_type':'ByDelimiter','delimiter':' ','lowercasing':'true'}],
            clf__dictionaries=[{'dictionary_id':'Word','gram_order':'1',
                'max_dictionary_size':str(spec.get('text_dictionary_size',12000)),
                'occurrence_lower_bound':str(spec.get('text_min_count',3))}],
            clf__feature_calcers=['BoW:top_tokens_count='+str(spec.get('bow_features',4000)),'NaiveBayes'])
    return model


def build_frames(work, config, experiment):
    cache = work/'cache'
    prepared = json.loads((cache/'prepared.json').read_text())
    assert verified_marker(cache/'prepared.json', prepared['fingerprint'])
    train, test = pd.read_parquet(cache/'train.parquet'), pd.read_parquet(cache/'test.parquet')
    assert 'label' not in test and 'class' not in test and not set(train.author) & set(test.author)
    assert train.groupby('author').fold.nunique().max() == 1
    assert len(train) == config['expected']['train'] and len(test) == config['expected']['test']
    posts, outcomes = pd.read_parquet(cache/'all_posts.parquet'), pd.read_parquet(cache/'history_outcomes.parquet')
    assert posts[KEYS].equals(outcomes[KEYS])
    numeric = json.loads((cache/'feature_columns.json').read_text())['numeric']
    columns, auxiliary, raw = [], [], {}
    for role, filename in [('text_encoder','text_embeddings'), ('image_encoder','image_embeddings')]:
        fp = identity({'settings':config[role], 'data':digest(cache/'all_posts.parquet')})
        assert verified_marker(cache/f'{filename}.json', fp)
        values = np.load(cache/f'{filename}.npy', allow_pickle=False)
        raw[role] = values
        if role == 'text_encoder':
            values = contextual_embeddings(posts, values, outcomes.reactions.to_numpy(), work)
            for i in range(8):
                key = f'rh_semantic_aux_{i}';posts[key] = values[:, -8+i];auxiliary.append(key)
        else:
            posts['rh_media_count'] = values[:, -1];auxiliary.append('rh_media_count')
        names = [f'{role}_component_{i}' for i in range(values.shape[1])]
        frame = pd.DataFrame(values, columns=names);frame[KEYS] = posts[KEYS]
        train = train.merge(frame, on=KEYS, validate='one_to_one');test = test.merge(frame, on=KEYS, validate='one_to_one')
        columns += names
    cutoffs = json.loads((cache/'label-audit.json').read_text()).get('cutoffs')
    if cutoffs is None:
        # Reproduce the fixed target definition using train authors' outcomes only.
        from kairos.label import fit_cutoffs
        assert json.loads((cache/'audit.json').read_text())['tied_author_timestamps'] == 0
        residuals = np.log1p(np.maximum(outcomes.reactions.to_numpy(float), 0)) - posts.level
        mask = posts.author.isin(set(train.author)) & posts.history_count.ge(5)
        cutoffs = fit_cutoffs(residuals[mask])
    extra = rich_history(posts, outcomes, cutoffs, work)
    retrieval = retrieval_history(posts, outcomes, raw['text_encoder'], raw['image_encoder'], cutoffs, work)
    extra = extra.merge(retrieval, on=KEYS, validate='one_to_one')
    extra = extra.merge(posts[KEYS+auxiliary], on=KEYS, validate='one_to_one')
    feature_file = experiment/'past-only-features.parquet'
    extra.to_parquet(feature_file, index=False)
    feature_sha = digest(feature_file)
    extra_columns = [c for c in extra if c not in KEYS]
    train = train.merge(extra, on=KEYS, validate='one_to_one');test = test.merge(extra, on=KEYS, validate='one_to_one')
    return train, test, numeric, extra_columns, columns, feature_sha


def load_predictions(work):
    predictions = {}
    for marker in sorted((work/'models').glob('*/complete.json')):
        state = json.loads(marker.read_text());assert verified_marker(marker,state['fingerprint'])
        with np.load(marker.parent/'probabilities.npz', allow_pickle=False) as values:
            predictions[marker.parent.name] = {k:values[k].copy() for k in ('oof','test')}
    return predictions


def publish_best(work, train, test, predictions):
    previous = json.loads((work/'selection.json').read_text())
    weights, scales, oof, values = base.choose_blend(sorted(predictions), predictions, train.label.to_numpy())
    selected = base.score(train.label,(oof*scales).argmax(1))
    if selected['macro_f1'] < previous['metrics']['macro_f1']:
        weights = previous['weights'];scales = np.array(previous['class_scales'])
        oof = sum(weights[n]*predictions[n]['oof'] for n in weights)
        values = sum(weights[n]*predictions[n]['test'] for n in weights)
        selected = base.score(train.label,(oof*scales).argmax(1))
    atomic_json(work/'selection.json',{'partition':'train_author_oof','weights':weights,
                'class_scales':scales.tolist(),'metrics':selected,
                'note':'Refinement and blend selection use train author OOF; this selection estimate is not untouched test.'})
    base.write_prediction(work/'predictions/final.csv',test,(values*scales).argmax(1))
    base.write_prediction(work/'predictions/best_available.csv',test,(values*scales).argmax(1))
    train[KEYS+['fold','label']].assign(prediction=(oof*scales).argmax(1)).to_csv(work/'predictions/oof.csv',index=False)
    path=work/'predictions/blend_probabilities.npz';tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as stream:np.savez_compressed(stream,oof=oof,test=values)
    tmp.replace(path)
    return oof,values,scales


def run(work, plan_path, group):
    stop_check(work)
    assert not (work/'report/official-test.json').exists(), 'Test already scored; scientific tuning refused'
    parent=json.loads((work/'environment/manifest.json').read_text());config=parent['config']
    root=Path(base.__file__).resolve().parents[1]
    assert parent['fingerprint']==identity({'code':artifacts_fingerprint(root),'config':config})
    plans=json.loads(plan_path.read_text())
    assert re.fullmatch(r'[a-z0-9-]{1,32}', group)
    assert 1 <= len(plans) <= 8 and len({s['name'] for s in plans}) == len(plans)
    assert all(s['name'].startswith(f'refine-{group}-') and re.fullmatch(r'[a-z0-9-]+',s['name'])
               and s['family']=='multimodal' for s in plans)
    experiment=work/'environment/refinements'/group;experiment.mkdir(parents=True,exist_ok=True)
    fingerprint=identity({'parent':parent['fingerprint'],'script':digest(Path(__file__)),'plan':plans})
    marker=experiment/'manifest.json'
    if marker.exists():assert json.loads(marker.read_text())['fingerprint']==fingerprint
    else:
        atomic_json(marker,{'at':now(),'fingerprint':fingerprint,'scientific_extension':True,'script_sha256':digest(Path(__file__)),
                    'parent_fingerprint':parent['fingerprint'],'config':config,'specs':plans,'test_targets_used':False,
                    'author_folds_preserved':True,'same_search_budget':json.loads((work/'search-budget.json').read_text())})
        shutil.copy2(work/'selection.json',experiment/'selection-before.json')
        shutil.copy2(work/'predictions/final.csv',experiment/'predictions-before.csv')
    train,test,numeric,extra,emb_columns,feature_sha=build_frames(work,config,experiment)
    predictions=load_predictions(work)
    budget_end=json.loads((work/'search-budget.json').read_text())['ends']
    if config.get('deadline'):
        budget_end=min(budget_end,datetime.fromisoformat(config['deadline']).timestamp()-config['final_buffer_hours']*3600)
    results=[];base.estimator=extended_estimator
    try:
        for requested in plans:
            stop_check(work)
            assert not (work/'report/official-test.json').exists()
            if time.time()>=budget_end:break
            spec=dict(requested);spec['extension_fingerprint']=fingerprint
            if spec.get('rich_history'):spec['feature_sha256']=feature_sha
            try:
                if spec.get('native_text'):
                    for frame in (train,test):
                        body = frame['text'].fillna('') + ' profile ' + frame['profile_text'].fillna('').str.slice(0,1500)
                        frame['native_text'] = native_text(body,spec.get('retain_entities',False))
                selected_embeddings = emb_columns
                if spec.get('embedding_mode') == 'current-text':
                    width = (sum(c.startswith('text_encoder_component_') for c in emb_columns)-8)//2
                    selected_embeddings = [f'text_encoder_component_{i}' for i in range(width)]
                value=base.evaluate_candidate(spec,config,train,test,numeric+(extra if spec.get('rich_history') else []),work,budget_end,selected_embeddings)
            except base.BudgetReached:break
            predictions[spec['name']]={k:value[k].copy() for k in ('oof','test')};value.close()
            oof,values,scales=publish_best(work,train,test,predictions)
            results.append({'candidate':spec['name'],'metrics':json.loads((work/'models'/spec['name']/'metrics.json').read_text())['oof'],
                            'selected_metrics':json.loads((work/'selection.json').read_text())['metrics']})
            atomic_json(experiment/'progress.json',{'at':now(),'trials':results,'best_selection':json.loads((work/'selection.json').read_text())})
            if results[-1]['selected_metrics']['macro_f1']>=.70:break
    finally:base.estimator=ORIGINAL_ESTIMATOR
    oof,values,scales=publish_best(work,train,test,predictions)
    report={'selection':json.loads((work/'selection.json').read_text()),
            'candidates':{n:json.loads((work/'models'/n/'metrics.json').read_text()) for n in predictions},
            'confusion_train_oof':confusion_matrix(train.label,(oof*scales).argmax(1),labels=[0,1,2]).tolist(),
            'per_class_train_oof':classification_report(train.label,(oof*scales).argmax(1),output_dict=True,zero_division=0),
            'scientific_extensions':[str(p.relative_to(work)) for p in sorted((work/'environment/refinements').glob('*/manifest.json'))]}
    report.update(base.diagnostics(train,(oof*scales).argmax(1),predictions['baseline']['oof'].argmax(1),config['bootstrap_samples'],config['seed']))
    atomic_json(work/'report/training.json',report)
    lines=['# Kairos: original models and authorized refinements','','Selection uses author-grouped train OOF only; official test remains unseen.','',
           '| Candidate | OOF macro-F1 | QWK |','| --- | --- | --- |']
    for name,result in report['candidates'].items():lines.append(f"| {name} | {result['oof']['macro_f1']:.4f} | {result['oof']['qwk']:.4f} |")
    lines+=['','Native lexical trials add CatBoost BoW/online NaiveBayes text features fitted inside each training fold. Current anonymized entity tokens are retained only in separately identified variants; no current outcomes or test targets enter text.',
            'OOF is reused for model/blend selection and may be optimistic. Snapshot features are retrospective, not a real-time deployment claim.',
            'Source/config/spec fingerprints, trials and immutable parent outputs are retained under environment/refinements.']
    (work/'report/REPORT.md').write_text('\n'.join(lines)+'\n')
    atomic_json(experiment/'complete.json',{'at':now(),'fingerprint':fingerprint,'trials':results,'best_selection':report['selection']})
    return report['selection']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--work',type=Path,default=Path('/work'))
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--group',required=True);args=parser.parse_args()
    atomic_json(args.work/'status.json',{'state':'running','stage':'train','extension':args.group,'at':now()})
    try:
        result=run(args.work,args.plan,args.group)
        atomic_json(args.work/'status.json',{'state':'completed','stage':'refinement','extension':args.group,'at':now(),'selection':result})
    except Exception as exc:
        atomic_json(args.work/'status.json',{'state':'needs_attention','stage':'refinement','extension':args.group,'at':now(),
                                            'error_type':type(exc).__name__,'reason':str(exc)})
        raise
