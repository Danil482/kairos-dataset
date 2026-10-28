import gc
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold,GroupShuffleSplit
from threadpoolctl import threadpool_limits
from solution.models import fit_model,probabilities,EmbeddedHistory,StackBundle,stack_features,classifier,calibrate


def load_frames(cache,configuration):
    frames={}
    for component in configuration['components']:
        name=component['cache']
        if name not in frames:
            frames[name]=pd.read_pickle(Path(cache)/(name+'.pkl'))
            table=frames[name]
            if table.duplicated(['author','post']).any() or table.loc[table.split.eq('test'),'label'].notna().any():
                raise ValueError('Feature keys must be unique and test targets masked')
    reference=next(iter(frames.values()))
    for table in frames.values():
        pd.testing.assert_frame_equal(reference[['author','post','split']],table[['author','post','split']])
    if set(reference.loc[reference.split.eq('train'),'author'])&set(reference.loc[reference.split.eq('test'),'author']):
        raise ValueError('Official author partitions overlap')
    return frames


def train(configuration,frames,model_dir):
    target=Path(model_dir)
    if (target/'model.joblib').exists():
        raise FileExistsError('Preserve the existing model; choose a new --model-dir for retraining')
    target.mkdir(parents=True,exist_ok=True)
    reference=next(iter(frames.values()))
    fitting=reference.loc[reference.split.eq('train')].copy()
    fitting['label']=fitting.label.astype(int)
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=configuration['seed']+1009)
        .split(fitting,groups=fitting.author))
    inner,cal=fitting.iloc[a],fitting.iloc[b]
    assert not(set(inner.author)&set(cal.author))
    cp,components=[],[]
    for number,item in enumerate(configuration['components']):
        schema,config=item['schema'],item['config'];table=frames[item['cache']]
        print('TRAIN_COMPONENT',number,config['kind'],flush=True)
        model=fit_model(config,schema,table.loc[inner.index],configuration['seed'])
        cp.append(probabilities(model,table.loc[cal.index],schema,config))
        del model;gc.collect()
        model=fit_model(config,schema,table.loc[fitting.index],configuration['seed'])
        if config['kind']=='history':model=EmbeddedHistory.from_model(model)
        components.append({**item,'model':model})
        joblib.dump(model,target/f'component-{number}.joblib',compress=3)
    x=stack_features(cp,cal,True);y=cal.label.to_numpy(dtype=int)
    oof=np.full((len(cal),3),np.nan)
    splitter=GroupKFold(n_splits=configuration['meta_folds'],shuffle=True,random_state=configuration['meta_seed'])
    protocol={'C':configuration['meta_C'],'max_iter':configuration['meta_max_iter']}
    with threadpool_limits(limits=4):
        for a,b in splitter.split(x,groups=cal.author):
            assert not(set(cal.author.iloc[a])&set(cal.author.iloc[b]))
            oof[b]=classifier(protocol).fit(x[a],y[a]).predict_proba(x[b])
        weights,score=calibrate(y,oof)
        meta=classifier(protocol).fit(x,y)
    bundle=StackBundle(components,meta,'context',weights)
    temporary=target/'model.joblib.tmp';joblib.dump(bundle,temporary,compress=3)
    temporary.replace(target/'model.joblib')
    return bundle,{'calibration_macro_f1':score,'decision_weights':weights,
        'fitting_posts':len(fitting),'fitting_authors':fitting.author.nunique(),
        'inner_authors':inner.author.nunique(),'calibration_authors':cal.author.nunique()}
