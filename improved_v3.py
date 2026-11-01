"""Small cached-feature refinement of v2; no encoder inference required.
Run: python improved_v3.py --stage validate
Then: python improved_v3.py --stage final
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score,confusion_matrix
from improved_v2 import tab,add_vectors,NUMERIC,EXTRA,key_digest
from improved import text_model
from kairos.metrics import score

PARAMS={
 'v2':{},
 'smooth':dict(max_leaf_nodes=15,l2_regularization=3.,learning_rate=.05,max_iter=600),
 'richer':dict(max_leaf_nodes=63,min_samples_leaf=30,l2_regularization=5.,learning_rate=.05,max_iter=600),
}

def fit_probabilities(tr,va,cols,emb,settings):
    tm=text_model().fit(tr.text,tr.label)
    text=tm.predict_proba(va.text)
    svd=TruncatedSVD(n_components=64,n_iter=5,random_state=42)
    a=svd.fit_transform(tm.named_steps['tfidf'].transform(tr.text))
    b=svd.transform(tm.named_steps['tfidf'].transform(va.text))
    norm=Normalizer();ta,names=add_vectors(tr,norm.fit_transform(a),'lsa');tb,_=add_vectors(va,norm.transform(b),'lsa')
    pca=PCA(n_components=64,random_state=42)
    ea,en=add_vectors(tr,pca.fit_transform(emb[tr.row_id]),'semantic')
    eb,_=add_vectors(va,pca.transform(emb[va.row_id]),'semantic')
    result={}
    for name in settings:
        probs=[]
        for trainx,valx,extra in [(ta,tb,names),(ea,eb,en)]:
            m=tab(cols+extra)
            m.named_steps['clf'].set_params(**PARAMS[name])
            m.fit(trainx,tr.label)
            probs.append(m.predict_proba(valx))
        result[name]=.85*(probs[0]+probs[1])/2+.15*text
        print(name,'unadjusted',score(va.label,result[name].argmax(1)) if 'label' in va else '',flush=True)
    for name in ['smooth','richer']:
        if name in result and 'v2' in result:
            result[f'blend_{name}']=(result[name]+result['v2'])/2
    return result

def decision(p,below,typical):
    return (p*np.array([below,typical,1.])).argmax(1)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=['validate','final'],required=True)
    ap.add_argument('--cache',type=Path,default=Path('results/v2_cache'))
    ap.add_argument('--output',type=Path,default=Path('lab1_submission_v3'))
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    df=pd.read_pickle(args.cache/'table.pkl');df['row_id']=np.arange(len(df))
    cols=NUMERIC+EXTRA+json.loads((args.cache/'context_columns.json').read_text())
    emb=np.load(args.cache/'embeddings.npy',mmap_mode='r')
    cfg=json.loads((args.cache/'embedding_config.json').read_text())
    assert cfg['key_sha256']==key_digest(df) and len(emb)==len(df)
    train=df[df.split=='train'];test=df[df.split=='test']
    assert not set(train.author)&set(test.author)
    if args.stage=='validate':
        reports=[]
        for seed in [42,123,2026]:
            ti,vi=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(train,groups=train.author))
            tr,va=train.iloc[ti],train.iloc[vi]
            assert not set(tr.author)&set(va.author)
            print('Validation seed',seed,flush=True)
            ps=fit_probabilities(tr,va,cols,emb,list(PARAMS))
            rows=[]
            for name,p in ps.items():
                for below in [.9,1.,1.1]:
                    for typical in [1.4,1.6,1.8]:
                        pred=decision(p,below,typical)
                        rows.append(dict(model=name,below_weight=below,typical_weight=typical,
                            macro_f1=float(f1_score(va.label,pred,average='macro'))))
            incumbent=float(f1_score(va.label,decision(ps['v2'],1.,1.6),average='macro'))
            report=dict(seed=seed,incumbent=incumbent,candidates=rows)
            reports.append(report)
            print('Incumbent',incumbent,'best',max(rows,key=lambda x:x['macro_f1']),flush=True)
            (args.output/f'validation_{seed}.json').write_text(json.dumps(report,indent=2))
        rows=pd.DataFrame([dict(seed=r['seed'],**c) for r in reports for c in r['candidates']])
        means=rows.groupby(['model','below_weight','typical_weight'],sort=False).macro_f1.mean()
        name,b,t=means.idxmax()
        best=dict(model=name,below_weight=float(b),typical_weight=float(t),mean_macro_f1=float(means.max()))
        chosen=[next(c['macro_f1'] for c in r['candidates'] if c['model']==name and c['below_weight']==b and c['typical_weight']==t) for r in reports]
        delta=[v-r['incumbent'] for v,r in zip(chosen,reports)]
        selection=dict(selected=best,incumbent_mean=float(np.mean([r['incumbent'] for r in reports])),
            improvement_by_split=delta,key_sha256=key_digest(df),parameters=PARAMS,
            test_evaluation_recommended=bool(min(delta)>0 and np.mean(delta)>=.002))
        (args.output/'selection.json').write_text(json.dumps(selection,indent=2))
        print('FROZEN SELECTION',selection,flush=True)
        return
    s=json.loads((args.output/'selection.json').read_text());best=s['selected']
    assert s['key_sha256']==key_digest(df)
    if not s['test_evaluation_recommended']:
        raise SystemExit('Validation improvement was insufficient; keep v2.')
    name=best['model'];settings=['v2',name.removeprefix('blend_')] if name.startswith('blend_') else [name]
    # Drop test labels before fitting/predicting. They are read only for final scoring below.
    ps=fit_probabilities(train,test.drop(columns=['label']),cols,emb,settings)
    pred=decision(ps[name],best['below_weight'],best['typical_weight'])
    test[['author','post']].assign(label=pred).to_csv(args.output/'predictions.csv',index=False)
    m=score(test.label,pred);m['f1_by_class']=f1_score(test.label,pred,labels=[0,1,2],average=None).tolist()
    m['confusion_matrix']=confusion_matrix(test.label,pred,labels=[0,1,2]).tolist()
    result=dict(selection=best,test=m,n_test=len(test))
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2))
    print('Final test',m,flush=True)
if __name__=='__main__':main()
