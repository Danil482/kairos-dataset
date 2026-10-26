"""Verify deployed frozen heads reproduce development probabilities before scoring test."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from scripts.stability_models import load_all
from scripts.visual_models import image_projection
from scripts.priority_models import ordinal_probabilities
from scripts.pilot_models import restrict_support
from scripts.semantic_embeddings import save_json,fingerprint
from scripts.evaluate_frozen_test import OUT,MODELS,blend_folds,freeze

def main():
    config=freeze()
    payload,table,semantic,visual,_,_,_=load_all()
    checks=[]
    with threadpool_limits(limits=6):
        for fold in (1,2,3):
            folder=MODELS/f'fold_{fold}'
            keys=pd.read_csv(folder/'predictions.csv')[['author','post']]
            sample=np.linspace(0,len(keys)-1,64).astype(int)
            # Match the original full-validation PCA multiplication; small float32
            # batches can round a component across a learned tree threshold.
            positions=keys.merge(table.reset_index()[['author','post','index']],on=['author','post'],validate='one_to_one')['index'].to_numpy()
            part=table.iloc[positions].reset_index(drop=True)
            sem=joblib.load(folder/'semantic_pca.joblib').transform(semantic[positions])
            vis=image_projection(joblib.load(folder/'visual_pca.joblib'),visual[positions],part.visual_available)
            matrix=pd.concat([part,pd.DataFrame(sem,columns=[f'semantic_pca_{i}' for i in range(64)]),pd.DataFrame(vis,columns=[f'visual_pca_{i}' for i in range(64)])],axis=1)
            for seed in range(3):
                schema=json.loads((folder/f'joint_s{seed}_schema.json').read_text())
                heads=[CatBoostClassifier().load_model(str(folder/f'joint_s{seed}_gt{h}.cbm')).predict_proba(matrix[schema['columns']])[:,1] for h in (0,1)]
                actual=restrict_support(ordinal_probabilities(*heads),part,payload['cutoffs'][0])
                expected=np.load(folder/f'joint_s{seed}_probabilities.npz')['validation']
                np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=1e-6)
                checks.append(dict(fold=fold,seed=seed,rows=len(keys),maximum_error=float(np.max(abs(actual-expected)))))
    saved=np.load(OUT/'probabilities.npz')
    expected=blend_folds([saved[f'fold_{i}'] for i in (1,2,3)],[config['rules'][str(i)] for i in (1,2,3)])
    np.testing.assert_allclose(expected,saved['combined'],atol=1e-12)
    predictions=pd.read_csv(OUT/'predictions.csv')
    np.testing.assert_array_equal(predictions.label,saved['combined'].argmax(axis=1))
    assert not predictions.duplicated(['author','post']).any()
    save_json(OUT/'inference_audit.json',dict(passed=True,checks=checks,classifiers=18,test_rows=len(predictions),prediction_sha256=fingerprint(OUT/'predictions.csv')))
    print('Frozen inference audit passed for all 18 heads and final test aggregation',flush=True)

if __name__=='__main__':main()
