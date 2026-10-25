import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from kairos_lab.common import artifacts_fingerprint,atomic_json,identity,mark
from kairos_lab.dataset import prepare
from kairos_lab.train import train_models
import kairos_lab.train as base
from tests.fixtures import make_release

path=Path(os.environ.get('KAIROS_REFINE_SCRIPT',Path(__file__).resolve().parents[1]/'scripts/refine-retrieval.py'))
spec=importlib.util.spec_from_file_location('authorized_refinement',path)
refine=importlib.util.module_from_spec(spec);spec.loader.exec_module(refine)


class RefinementTests(unittest.TestCase):
    def test_lagged_features_exclude_current_future_ties_and_other_authors(self):
        rows=[];outcomes=[]
        for author in ('A','B'):
            times=np.arange(12);times[9]=times[8]
            reactions=np.arange(1,13)**2;logs=np.log1p(reactions)
            for i in range(12):
                end=np.searchsorted(times,times[i],side='left')
                rows.append({'author':author,'post':f'{author}{i}','posted_at':pd.Timestamp('2020-01-01',tz='UTC')+pd.Timedelta(days=int(times[i])),
                             'level':np.median(logs[:end]) if end else np.nan,'history_count':end})
                outcomes.append({'author':author,'post':f'{author}{i}','reactions':int(reactions[i])})
        posts=pd.DataFrame(rows);original=pd.DataFrame(outcomes);changed=original.copy()
        changed.loc[8:,'reactions']=99999999
        first=refine.rich_history(posts,original,[-.55,1.15])
        second=refine.rich_history(posts,changed,[-.55,1.15])
        pd.testing.assert_frame_equal(first.iloc[:10],second.iloc[:10])

    def test_retrieval_excludes_current_future_ties_and_other_authors(self):
        rows=[];outcomes=[]
        for author in ('A','B'):
            times=np.arange(12);times[9]=times[8]
            logs=np.log1p(np.arange(1,13)**2)
            for i in range(12):
                end=np.searchsorted(times,times[i],side='left')
                rows.append({'author':author,'post':f'{author}{i}','posted_at':pd.Timestamp('2020-01-01',tz='UTC')+pd.Timedelta(days=int(times[i])),
                             'level':np.median(logs[:end]) if end else np.nan,'history_count':end})
                outcomes.append({'author':author,'post':f'{author}{i}','reactions':int(np.expm1(logs[i]))})
        posts=pd.DataFrame(rows);original=pd.DataFrame(outcomes)
        rng=np.random.default_rng(5);text=rng.normal(size=(24,8)).astype('float32')
        images=np.column_stack([rng.normal(size=(24,4)),np.ones(24)]).astype('float32')
        first=refine.retrieval_history(posts,original,text,images,[-.55,1.15])
        changed=original.copy();changed.loc[8:,'reactions']=9999999
        second=refine.retrieval_history(posts,changed,text,images,[-.55,1.15])
        pd.testing.assert_frame_equal(first.iloc[:10],second.iloc[:10])
        # Embeddings of later/tied posts and another author cannot influence query 8.
        text2=text.copy();images2=images.copy();text2[9:]*=-8;images2[9:,:-1]*=-8
        third=refine.retrieval_history(posts,original,text2,images2,[-.55,1.15])
        pd.testing.assert_frame_equal(first.iloc[:9],third.iloc[:9])
        changed=original.copy();changed.loc[:7,'reactions']=9999999
        fourth=refine.retrieval_history(posts,changed,text,images,[-.55,1.15])
        self.assertNotEqual(first.loc[8,'sr_text_10_centered_mean'],fourth.loc[8,'sr_text_10_centered_mean'])
        self.assertTrue(first.filter(regex='^sr_').iloc[0].isna().all())

    def test_full_extension_resume_and_test_target_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data';work=root/'work'
            config=make_release(data);config['enable_images']=True;config['bootstrap_samples']=3
            prepare(data,work,config)
            posts=pd.read_parquet(work/'cache/all_posts.parquet');rng=np.random.default_rng(7)
            for role,filename,width in [('text_encoder','text_embeddings',4),('image_encoder','image_embeddings',5)]:
                np.save(work/f'cache/{filename}.npy',rng.normal(size=(len(posts),width)).astype('float32'))
                from kairos_lab.common import digest
                fp=identity({'settings':config[role],'data':digest(work/'cache/all_posts.parquet')})
                mark(work/f'cache/{filename}.json',fp,[work/f'cache/{filename}.npy'])
            train_models(work,config)
            parent=identity({'code':artifacts_fingerprint(Path(base.__file__).resolve().parents[1]),'config':config})
            atomic_json(work/'environment/manifest.json',{'fingerprint':parent,'config':config})
            plan=root/'plan.json';plan.write_text(json.dumps([{'name':'refine-smoke-small','family':'multimodal',
                'parameters':{'depth':2,'iterations':8,'learning_rate':.2,'class_power':.5},'pca_components':12,
                'l2_leaf_reg':5,'rich_history':True,'embedding_mode':'current-text'}]))
            labels=pd.read_csv(data/'labels.csv');labels.loc[labels.split=='test','label']=2
            labels.to_csv(data/'labels.csv',index=False)
            selection=refine.run(work,plan,'smoke')
            first=(work/'predictions/final.csv').read_bytes()
            refine.run(work,plan,'smoke')
            self.assertEqual(first,(work/'predictions/final.csv').read_bytes())
            self.assertTrue((work/'models/refine-smoke-small/complete.json').exists())
            self.assertFalse((work/'report/official-test.json').exists())
            before=json.loads((work/'environment/refinements/smoke/selection-before.json').read_text())
            self.assertGreaterEqual(selection['metrics']['macro_f1'],before['metrics']['macro_f1'])


if __name__=='__main__':unittest.main()
