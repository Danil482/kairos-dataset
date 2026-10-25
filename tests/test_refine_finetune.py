import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaModel

import kairos_lab.train as base
from kairos_lab.common import artifacts_fingerprint, atomic_json, digest, identity, mark
from kairos_lab.dataset import prepare
from tests.fixtures import make_release

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(os.environ.get('KAIROS_FT_SCRIPT', ROOT/'scripts/refine-finetune.py'))
HELPER = Path(os.environ.get('KAIROS_FT_HELPER', ROOT/'scripts/refine-retrieval.py'))
UTILS = Path(os.environ.get('KAIROS_FT_UTILS', ROOT/'scripts/refine-neural.py'))
spec = importlib.util.spec_from_file_location('ft_extension', SCRIPT)
ft = importlib.util.module_from_spec(spec);spec.loader.exec_module(ft)
utils = ft.module(UTILS, 'test_ft_nn_utils')


def small_spec():
    return {'name': 'refine-smoke-finetune', 'family': 'finetune', 'last_layers': 1, 'epochs': 2,
            'batch_size': 4, 'max_length': 16, 'encoder_lr': .002, 'head_lr': .003, 'head_width': 12,
            'dropout': .2, 'weight_decay': .01, 'warmup_fraction': .1, 'class_power': .5,
            'gradient_clip': 1., 'precision': 'float32', 'checkpoint_steps': 2}


def tiny_snapshot(path):
    path.mkdir(parents=True)
    torch.manual_seed(7)
    config = XLMRobertaConfig(vocab_size=32, hidden_size=12, num_hidden_layers=2,
               num_attention_heads=3, intermediate_size=24, max_position_embeddings=64)
    XLMRobertaModel(config, add_pooling_layer=False).save_pretrained(path)
    vocab = {'<s>':0, '<pad>':1, '</s>':2, '<unk>':3, '<mask>':4,
             'hello':5, 'world':6, 'post':7, 'text':8, 'document':9, 'profile':10}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token='<unk>'));backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token='<s>', cls_token='<s>',
                 eos_token='</s>', sep_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
    tokenizer.save_pretrained(path)


class FineTuneTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory();self.root = Path(self.tmp.name)
        self.snapshot = self.root/'snapshot';tiny_snapshot(self.snapshot)

    def tearDown(self):self.tmp.cleanup()

    def test_mid_epoch_resume_restores_scheduler_shuffle_and_dropout(self):
        rng = np.random.default_rng(8)
        ids = rng.integers(5, 30, size=(24, 12), dtype=np.int32);masks = np.ones_like(ids, dtype=np.int8)
        numeric = rng.normal(size=(24, 5)).astype('float32');numeric[0,0] = np.nan
        labels = np.tile(np.arange(3),8);spec = small_spec()
        a, pa = ft.fit_encoder(ids,masks,numeric,labels,self.snapshot,spec,self.root/'a',self.root,'abc','cpu',42,float('inf'),utils)
        ft.fit_encoder(ids,masks,numeric,labels,self.snapshot,spec,self.root/'b',self.root,'abc','cpu',42,float('inf'),utils,until_step=3)
        b, pb = ft.fit_encoder(ids,masks,numeric,labels,self.snapshot,spec,self.root/'b',self.root,'abc','cpu',42,float('inf'),utils)
        np.testing.assert_array_equal(ft.predict_encoder(a,pa,ids,masks,numeric,'cpu',spec,utils),ft.predict_encoder(b,pb,ids,masks,numeric,'cpu',spec,utils))
        self.assertLessEqual(len(list((self.root/'b').glob('optimizer-step-*.pt'))),2)
        self.assertEqual(len(list((self.root/'b').glob('epoch-end-*.pt'))),2)
        self.assertTrue(all(not p.requires_grad for p in b.backbone.embeddings.parameters()))
        self.assertTrue(any(p.requires_grad for p in b.backbone.encoder.layer[-1].parameters()))

    def test_corrupt_mid_epoch_checkpoint_refuses_resume(self):
        ids = np.full((12,8),5,np.int32);masks = np.ones_like(ids,dtype=np.int8);numeric = np.zeros((12,4),np.float32);labels = np.tile(np.arange(3),4)
        ft.fit_encoder(ids,masks,numeric,labels,self.snapshot,small_spec(),self.root/'a',self.root,'abc','cpu',42,float('inf'),utils,until_step=1)
        state = json.loads((self.root/'a/latest.json').read_text());(self.root/'a'/state['checkpoint']).write_bytes(b'corrupt')
        with self.assertRaisesRegex(RuntimeError,'modified artifact'):
            ft.fit_encoder(ids,masks,numeric,labels,self.snapshot,small_spec(),self.root/'a',self.root,'abc','cpu',42,float('inf'),utils)

    def test_head_tail_tokens_and_training_only_normalizer(self):
        class Tokens:cls_token_id=0;sep_token_id=2;pad_token_id=1
        ids,mask = ft.pack_tokens(list(range(5,45)),Tokens(),16)
        self.assertEqual(mask.sum(),16);self.assertEqual(ids[0],0);self.assertEqual(ids[-1],2)
        self.assertTrue(np.array_equal(ids[-7:-1],np.arange(39,45)))
        x = np.arange(60,dtype='float32').reshape(12,5);before = utils.normalizer(x[:8]);x[8:]=99999
        after = utils.normalizer(x[:8])
        for key in before:np.testing.assert_array_equal(before[key],after[key])

    def test_complete_author_oof_run_resume_and_test_target_isolation(self):
        data,work,models_root = self.root/'data',self.root/'work',self.root/'models'
        config = make_release(data);config.update(enable_images=True,bootstrap_samples=3)
        prepare(data,work,config);posts=pd.read_parquet(work/'cache/all_posts.parquet');rng=np.random.default_rng(7)
        for role,name,width in [('text_encoder','text_embeddings',4),('image_encoder','image_embeddings',5)]:
            np.save(work/f'cache/{name}.npy',rng.normal(size=(len(posts),width)).astype('float32'))
            fp=identity({'settings':config[role],'data':digest(work/'cache/all_posts.parquet')});mark(work/f'cache/{name}.json',fp,[work/f'cache/{name}.npy'])
        base.train_models(work,config)
        parent=identity({'code':artifacts_fingerprint(Path(base.__file__).resolve().parents[1]),'config':config});atomic_json(work/'environment/manifest.json',{'fingerprint':parent,'config':config})
        snapshot=models_root/'text_encoder'/config['text_encoder']['revision']
        shutil.copytree(self.snapshot,snapshot)
        mark(snapshot/'verified.json',identity(config['text_encoder']),sorted(p for p in snapshot.iterdir() if p.is_file()))
        posts[['author','post']].assign(rh_past_level=posts.level).to_parquet(work/'cache/test-past.parquet',index=False)
        plan=self.root/'plan.json';plan.write_text(json.dumps({'past_features':'cache/test-past.parquet','past_features_sha256':digest(work/'cache/test-past.parquet'),'candidates':[small_spec()]}))
        labels=pd.read_csv(data/'labels.csv');labels.loc[labels.split=='test','label']=2;labels.to_csv(data/'labels.csv',index=False)
        selected=ft.run(work,models_root,plan,'smoke',HELPER,UTILS,device='cpu')
        first=(work/'predictions/final.csv').read_bytes();ft.run(work,models_root,plan,'smoke',HELPER,UTILS,device='cpu')
        self.assertEqual(first,(work/'predictions/final.csv').read_bytes());self.assertFalse((work/'report/official-test.json').exists())
        self.assertTrue((work/'models/refine-smoke-finetune/complete.json').exists())
        before=json.loads((work/'environment/refinements/smoke/selection-before.json').read_text());self.assertGreaterEqual(selected['metrics']['macro_f1'],before['metrics']['macro_f1'])


if __name__ == '__main__':unittest.main()
