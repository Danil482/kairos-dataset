import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import kairos_lab.train as base
from kairos_lab.common import artifacts_fingerprint, atomic_json, digest, identity, mark
from kairos_lab.dataset import prepare
from tests.fixtures import make_release

SCRIPT = Path(os.environ.get('KAIROS_NEURAL_SCRIPT', Path(__file__).resolve().parents[1]/'scripts/refine-neural.py'))
HELPER = Path(os.environ.get('KAIROS_NEURAL_HELPER', Path(__file__).resolve().parents[1]/'scripts/refine-retrieval.py'))
spec = importlib.util.spec_from_file_location('neural_extension', SCRIPT)
neural = importlib.util.module_from_spec(spec);spec.loader.exec_module(neural)


def small_spec():
    return {'name': 'refine-smoke-neural', 'family': 'neural', 'hidden': [16, 8], 'dropout': .2,
            'epochs': 5, 'swa_last_epochs': 2, 'learning_rate': .003,
            'weight_decay': .01, 'class_power': .5, 'batch_size': 16}


class NeuralTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory();self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_resume_preserves_optimizer_shuffle_and_dropout_rng(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(96, 12)).astype('float32');x[0, 0] = np.nan
        y = np.tile(np.arange(3), 32)
        a, pa = neural.fit_network(x, y, small_spec(), self.root/'a', self.root, 'abc', 'cpu', 42, float('inf'))
        neural.fit_network(x, y, small_spec(), self.root/'b', self.root, 'abc', 'cpu', 42, float('inf'), until_epoch=3)
        b, pb = neural.fit_network(x, y, small_spec(), self.root/'b', self.root, 'abc', 'cpu', 42, float('inf'))
        np.testing.assert_array_equal(neural.predict(a, pa, x, 'cpu'), neural.predict(b, pb, x, 'cpu'))
        self.assertEqual(json.loads((self.root/'b/latest.json').read_text())['epoch'], 5)

    def test_corrupt_checkpoint_refuses_resume(self):
        x = np.zeros((30, 5), dtype='float32');y = np.tile(np.arange(3), 10)
        neural.fit_network(x, y, small_spec(), self.root/'a', self.root, 'abc', 'cpu', 42, float('inf'), until_epoch=1)
        (self.root/'a/optimizer-epoch-001.pt').write_bytes(b'corrupt')
        with self.assertRaisesRegex(RuntimeError, 'modified artifact'):
            neural.fit_network(x, y, small_spec(), self.root/'a', self.root, 'abc', 'cpu', 42, float('inf'))

    def test_validation_rows_do_not_fit_normalization(self):
        x = np.arange(60, dtype='float32').reshape(12, 5)
        first = neural.normalizer(x[:8])
        changed = x.copy();changed[8:] = 999999
        second = neural.normalizer(changed[:8])
        for key in first:np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_array_equal(neural.normalize(x[:8], first), neural.normalize(changed[:8], second))

    def test_complete_author_oof_run_resume_and_test_target_isolation(self):
        data, work = self.root/'data', self.root/'work'
        config = make_release(data);config.update(enable_images=True, bootstrap_samples=3)
        prepare(data, work, config)
        posts = pd.read_parquet(work/'cache/all_posts.parquet');rng = np.random.default_rng(7)
        for role, filename, width in [('text_encoder', 'text_embeddings', 4), ('image_encoder', 'image_embeddings', 5)]:
            np.save(work/f'cache/{filename}.npy', rng.normal(size=(len(posts), width)).astype('float32'))
            fp = identity({'settings': config[role], 'data': digest(work/'cache/all_posts.parquet')})
            mark(work/f'cache/{filename}.json', fp, [work/f'cache/{filename}.npy'])
        base.train_models(work, config)
        parent = identity({'code': artifacts_fingerprint(Path(base.__file__).resolve().parents[1]), 'config': config})
        atomic_json(work/'environment/manifest.json', {'fingerprint': parent, 'config': config})
        past = posts[['author', 'post']].assign(rh_past_level=posts.level)
        past.to_parquet(work/'cache/test-past.parquet', index=False)
        plan = self.root/'plan.json'
        plan.write_text(json.dumps({'past_features': 'cache/test-past.parquet',
                        'past_features_sha256': digest(work/'cache/test-past.parquet'), 'candidates': [small_spec()]}))
        labels = pd.read_csv(data/'labels.csv');labels.loc[labels.split == 'test', 'label'] = 2
        labels.to_csv(data/'labels.csv', index=False)
        selection = neural.run(work, plan, 'smoke', HELPER, device='cpu')
        first = (work/'predictions/final.csv').read_bytes()
        neural.run(work, plan, 'smoke', HELPER, device='cpu')
        self.assertEqual(first, (work/'predictions/final.csv').read_bytes())
        self.assertTrue((work/'models/refine-smoke-neural/complete.json').exists())
        self.assertFalse((work/'report/official-test.json').exists())
        before = json.loads((work/'environment/refinements/smoke/selection-before.json').read_text())
        self.assertGreaterEqual(selection['metrics']['macro_f1'], before['metrics']['macro_f1'])


if __name__ == '__main__':
    unittest.main()
