"""Freeze existing development models, infer official test, then score once.

No fitting or test-driven selection. Test labels are read only by score_saved().
"""
from pathlib import Path
import sys,json,shutil,argparse
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from student_features import build_features,normalize_text
from scripts.analyze_train import extra_content,lag_features,as_dict
from scripts.semantic_embeddings import collect_texts,save_json,fingerprint
from scripts.semantic_history import history_features,encode_missing
from scripts.priority_features import contextual_history,content_fields
from scripts.priority_models import ordinal_probabilities
from scripts.visual_models import image_projection
from scripts.pilot_models import restrict_support
from train_maria import metrics,bootstrap_authors

OUT=ROOT/'results'/'official_test_v3'
MODELS=ROOT/'results'/'stability_research'

def blend_folds(probabilities,multipliers):
    adjusted=[]
    for p,m in zip(probabilities,multipliers,strict=True):
        p=np.asarray(p,dtype=float); m=np.asarray(m,dtype=float)
        if p.ndim!=2 or p.shape[1]!=3 or m.shape!=(3,) or not np.isfinite(p).all() or (p<0).any() or (m<=0).any():
            raise ValueError('Invalid fold probabilities/rule')
        q=p*m
        adjusted.append(q/q.sum(axis=1,keepdims=True))
    if not adjusted: raise ValueError('No folds')
    return np.mean(adjusted,axis=0)

def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    files={}; rules={}
    for fold in (1,2,3):
        folder=MODELS/f'fold_{fold}'
        report=json.loads((folder/'report.json').read_text())
        rules[str(fold)]=report['models']['joint_mean_bootstrap_fine']['calibration']['multipliers']
        paths=[folder/'semantic_pca.joblib',folder/'visual_pca.joblib',folder/'report.json']
        paths += [folder/f'joint_s{s}_schema.json' for s in range(3)]
        paths += [folder/f'joint_s{s}_gt{h}.cbm' for s in range(3) for h in (0,1)]
        for p in paths: files[p.relative_to(ROOT).as_posix()]=fingerprint(p)
    config=dict(version=1,chosen='joint_mean_bootstrap_fine',selection='development OOF only',
        seed_combination='equal mean of three ordinal probability arrays within each fold',
        fold_combination='multiply each fold mean by its frozen inner-author rule, normalize rows, equal mean of three folds, argmax',
        retrained=False,classifiers=18,official_test_used_for_selection=False,rules=rules,files=files)
    path=OUT/'frozen_protocol.json'
    if path.exists():
        if json.loads(path.read_text())['config']!=config: raise ValueError('Frozen candidate changed')
    else:save_json(path,dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),config=config))
    return config

def raw_posts(authors):
    rows=[]
    for author in sorted(authors):
        feed=json.loads((ROOT/'data'/'posts'/f'{author}.json').read_text(encoding='utf-8'))
        for post in feed['posts']:
            if as_dict(post.get('author')).get('code')!=author:continue
            rows.append(dict(author=author,post=post['urn'],own=True,posted_at=post.get('postedDateTimestamp'),
                crawled_at=feed.get('crawledAt'),reactions=int(post.get('totalReactionCount') or 0),
                text=post.get('text') or '',content_type=post.get('contentType') or 'unknown',quote_mode=content_fields(post)['quote_mode']))
    frame=pd.DataFrame(rows)
    frame.posted_at=pd.to_datetime(frame.posted_at,unit='ms',utc=True)
    frame.crawled_at=pd.to_datetime(frame.crawled_at,utc=True,errors='coerce')
    return frame

def tabular(keys,cutoffs):
    posts=raw_posts(set(keys.author))
    base,_=build_features(posts,low=cutoffs[0],high=cutoffs[1])
    extra=extra_content(set(keys.author))
    table=keys.merge(base,on=['author','post'],validate='one_to_one').merge(extra,on=['author','post'],validate='one_to_one')
    table=table.merge(lag_features(posts),on=['author','post'],validate='one_to_one')
    corpus=posts.merge(extra[['author','post','expanded_text']],on=['author','post'],validate='one_to_one')
    corpus.text=corpus.text.map(normalize_text)
    last=keys.merge(posts[['author','post','posted_at']],on=['author','post'],validate='one_to_one').groupby('author').posted_at.max()
    corpus=corpus[corpus.posted_at.le(corpus.author.map(last))].reset_index(drop=True)
    positions=keys.merge(corpus.reset_index()[['author','post','index']],on=['author','post'],validate='one_to_one')['index'].to_numpy()
    return table,corpus,positions

def history_blocks(corpus,own,expanded,positions,cutoffs):
    blocks=[history_features(corpus,own,'own',cutoffs),history_features(corpus,expanded,'expanded',cutoffs)]
    quote=contextual_history(corpus,expanded,'quote','history_quote',cutoffs)
    quote['history_quote_mode']=corpus.quote_mode.to_numpy()
    return pd.concat([*blocks,quote],axis=1).iloc[positions].reset_index(drop=True)

def audit_training_features(payload):
    old=payload['table']; authors=sorted(old.author.unique())[:3]
    reference=old[old.author.isin(authors)].reset_index(drop=True)
    table,corpus,positions=tabular(reference[['author','post']],payload['cutoffs'])
    columns=payload['numeric']+payload['extra_numeric']+['text','expanded_text','content_type','writing_system']
    pd.testing.assert_frame_equal(table[columns],reference[columns],check_dtype=False)
    stored=joblib.load(ROOT/'results'/'semantic_history'/'corpus.joblib')
    indices=corpus[['author','post']].merge(stored.reset_index()[['author','post','index']],on=['author','post'],validate='one_to_one')['index'].to_numpy()
    v=np.vstack([np.zeros((1,768),np.float32),np.load(ROOT/'results'/'semantic_history'/'unique_vectors.npy')])
    mapping=np.load(ROOT/'results'/'semantic_history'/'corpus_vector_indices.npz')
    block=history_blocks(corpus,v[mapping['text'][indices]+1],v[mapping['expanded_text'][indices]+1],positions,payload['cutoffs'])
    hist=joblib.load(ROOT/'results'/'semantic_history'/'features.joblib')
    priority=joblib.load(ROOT/'results'/'priority_research'/'features.joblib')
    expected=reference[['author','post']].merge(hist,on=['author','post'],validate='one_to_one').merge(priority,on=['author','post'],validate='one_to_one')
    np.testing.assert_allclose(block.to_numpy(),expected[block.columns].to_numpy(),equal_nan=True,atol=1e-6)
    save_json(OUT/'training_feature_audit.json',dict(passed=True,authors=len(authors),posts=len(reference),columns=len(columns)+len(block.columns)))
    print('Training feature reproduction passed',flush=True)

def prepare():
    payload=joblib.load(ROOT/'results'/'exploration_v2'/'train_table.joblib')
    if not (OUT/'training_feature_audit.json').exists():audit_training_features(payload)
    path=OUT/'inputs.joblib'
    if path.exists():return joblib.load(path)
    keys=pd.read_csv(ROOT/'data'/'labels.csv',usecols=['author','post','split'])
    keys=keys[keys.split.eq('test')].drop(columns='split').reset_index(drop=True)
    assert len(keys)==17623 and keys.author.nunique()==824
    assert not set(keys.author)&set(payload['table'].author)
    print('Building label-blind test features',flush=True)
    table,corpus,positions=tabular(keys,payload['cutoffs'])
    texts,mapping,_=collect_texts(corpus)
    vectors_path=OUT/'text_vectors.npy'
    if vectors_path.exists():vectors=np.load(vectors_path)
    else:
        old_corpus=joblib.load(ROOT/'results'/'semantic_history'/'corpus.joblib')
        old_texts,_,_=collect_texts(old_corpus)
        lookup={text:i for i,text in enumerate(old_texts)}
        old_vectors=np.load(ROOT/'results'/'semantic_history'/'unique_vectors.npy',mmap_mode='r')
        vectors=np.zeros((len(texts),768),np.float32);missing=[]
        for i,text in enumerate(texts):
            if text in lookup:vectors[i]=old_vectors[lookup[text]]
            else:missing.append(i)
        if missing:vectors[missing]=encode_missing([texts[i] for i in missing],cache_dir=OUT/'e5')
        np.save(vectors_path,vectors)
    if vectors.shape!=(len(texts),768):raise ValueError('Text cache mismatch')
    vectors=np.vstack([np.zeros((1,768),np.float32),vectors])
    own=vectors[mapping['text']+1];expanded=vectors[mapping['expanded_text']+1]
    block=history_blocks(corpus,own,expanded,positions,payload['cutoffs'])
    table=pd.concat([table,block],axis=1)
    semantic=np.concatenate([own[positions],expanded[positions]],axis=1)
    import scripts.visual_embeddings as ve
    visual_dir=OUT/'visual';visual_dir.mkdir(exist_ok=True)
    with (ROOT/'data'/'media'/'manifest.jsonl').open(encoding='utf-8') as stream:
        pairs,excluded=ve.select_content_rows((json.loads(line) for line in stream),set(zip(keys.author,keys.post)))
    files=sorted({p['file'] for p in pairs});lookup={file:i for i,file in enumerate(files)}
    for pair in pairs:pair['image_index']=lookup[pair['file']]
    index=dict(config=dict(scope='official test label-blind inference',keys_sha256=__import__('hashlib').sha256(keys.to_csv(index=False).encode()).hexdigest()),files=files,pairs=pairs,excluded_rows=excluded)
    save_json(visual_dir/'media_index.json',index)
    shutil.copy2(ROOT/'results'/'visual'/'encoder.json',visual_dir/'encoder.json')
    previous=ve.OUT
    try:
        ve.OUT=visual_dir
        ve.encode(64)
    finally:ve.OUT=previous
    mean,first,metadata=ve.aggregate_posts(table,index,np.load(visual_dir/'image_vectors.npy',mmap_mode='r'),np.load(visual_dir/'image_details.npy',mmap_mode='r'))
    table=pd.concat([table,metadata],axis=1)
    assert 'label' not in table and 'reactions' not in table
    value=dict(table=table,semantic=semantic,visual=np.concatenate([mean,first],axis=1),cutoffs=payload['cutoffs'])
    joblib.dump(value,path,compress=3)
    return value

def predict(config):
    path=OUT/'predictions.csv'
    if path.exists():
        saved=json.loads((OUT/'prediction_manifest.json').read_text())
        if saved['sha256']!=fingerprint(path):raise ValueError('Predictions changed')
        return
    value=prepare();table=value['table'];folds=[]
    for fold in (1,2,3):
        folder=MODELS/f'fold_{fold}'
        sem=joblib.load(folder/'semantic_pca.joblib').transform(value['semantic'])
        vis=image_projection(joblib.load(folder/'visual_pca.joblib'),value['visual'],table.visual_available)
        matrix=pd.concat([table,pd.DataFrame(sem,columns=[f'semantic_pca_{i}' for i in range(64)]),pd.DataFrame(vis,columns=[f'visual_pca_{i}' for i in range(64)])],axis=1)
        seeds=[]
        for seed in range(3):
            schema=json.loads((folder/f'joint_s{seed}_schema.json').read_text())
            x=matrix[schema['columns']]
            assert not set(x.columns)&{'label','reactions','resid','author','post','split'}
            heads=[CatBoostClassifier().load_model(str(folder/f'joint_s{seed}_gt{h}.cbm')).predict_proba(x)[:,1] for h in (0,1)]
            seeds.append(restrict_support(ordinal_probabilities(*heads),table,value['cutoffs'][0]))
        folds.append(np.mean(seeds,axis=0))
        print(f'Predicted test with frozen fold {fold}',flush=True)
    probability=blend_folds(folds,[config['rules'][str(f)] for f in (1,2,3)])
    table[['author','post']].assign(label=probability.argmax(axis=1)).to_csv(path,index=False)
    np.savez_compressed(OUT/'probabilities.npz',combined=probability,**{f'fold_{i+1}':p for i,p in enumerate(folds)})
    save_json(OUT/'prediction_manifest.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),sha256=fingerprint(path),
        inputs_sha256=fingerprint(OUT/'inputs.joblib'),protocol_sha256=fingerprint(OUT/'frozen_protocol.json'),rows=len(table),test_labels_read=False))

def score_saved():
    target=OUT/'test_metrics.json'
    if target.exists():
        print('Official evaluation already saved; not selecting or retraining',flush=True);return
    predicted=pd.read_csv(OUT/'predictions.csv').rename(columns={'label':'predicted'})
    labels=pd.read_csv(ROOT/'data'/'labels.csv');labels=labels[labels.split.eq('test')]
    data=labels.merge(predicted,on=['author','post'],how='outer',validate='one_to_one',indicator=True)
    if not data._merge.eq('both').all():raise ValueError('Test prediction coverage mismatch')
    result=metrics(data.label,data.predicted)
    result.update(author_bootstrap_95=bootstrap_authors(data,'predicted'),rows=len(data),authors=int(data.author.nunique()),
        prediction_sha256=fingerprint(OUT/'predictions.csv'),protocol_sha256=fingerprint(OUT/'frozen_protocol.json'),
        evaluated_at_utc=datetime.now(timezone.utc).isoformat(),scope='Official test; frozen existing 18-head ensemble, no retraining or tuning on test')
    save_json(target,result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--predict-only',action='store_true');args=parser.parse_args()
    with threadpool_limits(limits=4):
        config=freeze();predict(config)
        if not args.predict_only:score_saved()
