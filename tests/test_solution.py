"""Portable solution checks with real tiny encoders and synthetic private-free data."""
import copy
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel

from kairos_lab.common import digest, identity, mark
from solution import pipeline as p
from tests.fixtures import make_release
from tests.test_refine_finetune import tiny_snapshot


def fixture(root):
    data, weights = root / 'data', root / 'weights'
    cfg = copy.deepcopy(p.settings())
    config = make_release(data)
    config.update(enable_images=True, threads=1)
    config['text_encoder'].update(repo='synthetic-xlm-roberta', revision='fixture',
                                  max_length=16, max_chunks=2, batch_size=2)
    config['image_encoder'].update(repo='synthetic-clip', revision='fixture', batch_size=2)
    cfg['config'] = config
    cfg['reference_prediction_sha256'] = None
    cfg['specs']['history-1']['parameters'] = dict(depth=2, iterations=6, learning_rate=.2, class_power=.5)
    cfg['specs'][p.TREE].update(pca_components=3, l2_leaf_reg=5)
    cfg['specs'][p.TREE]['parameters'] = dict(depth=2, iterations=6, learning_rate=.2, class_power=.5)
    for name in (p.TEACHER, p.FUSION):
        cfg['specs'][name].update(last_layers=1, epochs=1, batch_size=4, max_length=16,
                                  head_width=12, checkpoint_steps=10, precision='float32')
    text = weights / 'pretrained/text_encoder/fixture'
    tiny_snapshot(text)
    mark(text / 'verified.json', identity(config['text_encoder']), list(text.iterdir()))
    image = weights / 'pretrained/image_encoder/fixture'
    image.mkdir(parents=True)
    torch.manual_seed(7)
    model = CLIPModel(CLIPConfig(text_config=dict(vocab_size=32, hidden_size=8, intermediate_size=16,
                     num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=16),
                     vision_config=dict(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                     num_attention_heads=2, image_size=16, patch_size=8), projection_dim=8))
    model.save_pretrained(image)
    CLIPImageProcessor(size={'shortest_edge': 16}, crop_size={'height': 16, 'width': 16}).save_pretrained(image)
    mark(image / 'verified.json', identity(config['image_encoder']), list(image.iterdir()))
    Image.new('RGB', (20, 30), color='red').save(data / 'media/test.png')
    (data / 'media/manifest.jsonl').write_text(json.dumps(dict(author='AUTHOR_0000',
                  post='POST_0000_0005', path='media/test.png'))+'\n')
    return data, weights, cfg


class PortableSolutionTests(unittest.TestCase):
    def test_fresh_training_and_two_independent_inference_runs_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, weights, cfg = fixture(root)
            p.train(data, weights, root / 'training', cfg, 'cpu')
            original = (data / 'labels.csv').read_bytes()
            a = p.predict(data, weights, root / 'fresh-a', root / 'a.csv', cfg, 'cpu')
            b = p.predict(data, weights, root / 'fresh-b', root / 'b.csv', cfg, 'cpu')
            self.assertEqual((root / 'a.csv').read_bytes(), (root / 'b.csv').read_bytes())
            self.assertEqual(a['rows'], cfg['config']['expected']['test'])
            self.assertFalse(a['cached_prediction_files_read'])
            self.assertEqual(a['feature_sha256'], b['feature_sha256'])
            self.assertEqual(original, (data / 'labels.csv').read_bytes())
            with np.load(root / 'fresh-a/component-probabilities.npz') as x, np.load(root / 'fresh-b/component-probabilities.npz') as y:
                for name in x.files:
                    np.testing.assert_array_equal(x[name], y[name])
            frame = pd.read_csv(root / 'a.csv')
            self.assertFalse(frame.duplicated(['author', 'post']).any())
            self.assertTrue(frame.label.isin([0, 1, 2]).all())
            manifest = json.loads((weights / 'manifest.json').read_text())
            victim = weights / 'models/baseline/model.joblib'
            victim.write_bytes(victim.read_bytes()+b'corrupt')
            with self.assertRaisesRegex(ValueError, 'modified weight'):
                p.predict(data, weights, root / 'refused', root / 'refused.csv', cfg, 'cpu')
            self.assertFalse((root / 'refused').exists())
            self.assertNotEqual(digest(victim), manifest['files']['models/baseline/model.joblib'])

    def test_missing_dependency_model_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, weights, cfg = fixture(root)
            p.weight_manifest(weights, cfg, 'fixed-full-training')
            with self.assertRaisesRegex(ValueError, 'Incomplete ensemble'):
                p.verify_weights(weights, cfg)

    def test_weight_traversal_is_refused_before_loading_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, weights, cfg = fixture(root)
            manifest = {'configuration_sha256': identity(cfg), 'origin': 'fixed-full-training',
                        'files': {'../outside': '0'*64}}
            (weights / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Unsafe weight path'):
                p.verify_weights(weights, cfg)

    def test_docker_wrapper_separates_network_fetch_and_readonly_inference(self):
        with tempfile.TemporaryDirectory(prefix='kairos space ') as directory:
            root = Path(directory)
            (root / 'data').mkdir()
            fake = root / 'docker'
            fake.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            fake.chmod(0o755)
            env = {**os.environ, 'PATH': str(root)+os.pathsep+os.environ['PATH'],
                   'DATA': str(root / 'data'), 'WEIGHTS': str(root / 'weights'),
                   'WORK': str(root / 'work'), 'GPU_UUID': 'GPU-01234567-abcd-abcd-abcd-0123456789ab'}
            script = p.ROOT / 'run.sh'
            def run(command):
                return json.loads(subprocess.check_output(['bash', str(script), command], env=env, text=True))
            inference = run('predict')
            self.assertEqual(inference[inference.index('--gpus')+1], 'device='+env['GPU_UUID'])
            self.assertEqual(inference[inference.index('--network')+1], 'none')
            self.assertIn('type=bind,src='+str(Path(env['WEIGHTS']).resolve())+',dst=/weights,readonly', inference)
            self.assertIn('type=bind,src='+str(Path(env['DATA']).resolve())+',dst=/inputs,readonly', inference)
            fetch = run('fetch')
            self.assertNotIn('--gpus', fetch)
            self.assertFalse(any('dst=/inputs' in arg for arg in fetch))
            self.assertEqual(fetch[fetch.index('--network')+1], 'bridge')
            self.assertIn('type=bind,src='+str(Path(env['WEIGHTS']).resolve())+',dst=/weights', fetch)

    def test_docker_wrapper_refuses_gpu_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {**os.environ, 'DATA': directory, 'WEIGHTS': directory, 'WORK': directory, 'GPU_UUID': 'all'}
            result = subprocess.run(['bash', str(p.ROOT / 'run.sh'), 'predict'], env=env,
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('one GPU UUID', result.stderr)


if __name__ == '__main__':
    unittest.main()
