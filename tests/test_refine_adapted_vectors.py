import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import kairos_lab.train as base
from kairos_lab.common import artifacts_fingerprint, atomic_json, digest, identity, mark
from kairos_lab.dataset import prepare
from kairos_lab.encoders import contextual_embeddings, save_array
from tests.fixtures import make_release

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(os.environ.get('KAIROS_ADAPTED_SCRIPT', ROOT / 'scripts/refine-adapted-vectors.py'))
HELPER = Path(os.environ.get('KAIROS_ADAPTED_NATIVE', ROOT / 'scripts/refine-native-text.py'))
FTHELPER = Path(os.environ.get('KAIROS_ADAPTED_FT', ROOT / 'scripts/refine-finetune.py'))
spec = importlib.util.spec_from_file_location('adapted_extension', SCRIPT)
adapted = importlib.util.module_from_spec(spec);spec.loader.exec_module(adapted)


class AdaptedVectorTests(unittest.TestCase):
    def test_r05_backbone_weight_layout_loads_strictly_in_pinned_transformers(self):
        import torch
        from transformers import XLMRobertaConfig, XLMRobertaModel
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = XLMRobertaConfig(vocab_size=32, hidden_size=12, num_hidden_layers=2,
                                     num_attention_heads=3, intermediate_size=24, max_position_embeddings=32)
            original = XLMRobertaModel(config, add_pooling_layer=False);original.save_pretrained(path)
            state = {'weights': {'backbone.' + k: v.clone() for k, v in original.state_dict().items()}}
            state['weights']['head.0.weight'] = torch.ones(4, 12)
            loaded = adapted.load_backbone(path, state, 12)
            for key, value in original.state_dict().items():self.assertTrue(torch.equal(value, loaded.state_dict()[key]))
            with self.assertRaises(AssertionError):adapted.load_backbone(path, state, 24)
            state['weights'].pop(next(k for k in state['weights'] if k.startswith('backbone.')))
            with self.assertRaises(RuntimeError):adapted.load_backbone(path, state, 12)

    def test_fold_weights_never_fall_back_to_full_and_corruption_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory);source = work / 'environment/refinements/r05';source.mkdir(parents=True)
            trial = {'name': 'source-model', 'max_length': 16}
            atomic_json(source / 'manifest.json', {'fingerprint': 'source', 'plan': {'candidates': [trial]}})
            folder = work / 'models/source-model';folder.mkdir(parents=True)
            parent = identity({'parent': 'source', 'spec': trial})
            for job in ('fold-0', 'fold-1', 'fold-2', 'full'):
                path = folder / ('model.pt' if job == 'full' else job + '.pt');path.write_bytes(job.encode())
                fp = parent if job == 'full' else identity({'parent': parent, 'fold': int(job.split('-')[1])})
                mark(folder / ('complete.json' if job == 'full' else job + '.json'), fp, [path])
            plan = {'source_group': 'r05', 'source_group_fingerprint': 'source', 'source_candidate': 'source-model', 'max_length': 16}
            _, records = adapted.source_records(work, plan, 3)
            self.assertTrue(records['fold-0']['path'].endswith('/fold-0.pt'))
            self.assertTrue(records['full']['path'].endswith('/model.pt'))
            (folder / 'fold-1.json').unlink()
            with self.assertRaisesRegex(AssertionError, 'not an OOF fallback'):
                adapted.source_records(work, plan, 3)
            mark(folder / 'fold-1.json', identity({'parent': parent, 'fold': 1}), [folder / 'fold-1.pt'])
            (folder / 'fold-1.pt').write_bytes(b'corrupt')
            with self.assertRaisesRegex(RuntimeError, 'modified artifact'):
                adapted.source_records(work, plan, 3)

    def test_chunk_resume_integrity_and_row_order(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory);folder = work / 'chunks';calls = []
            expected = np.arange(39, dtype=np.float32).reshape(13, 3)
            def compute(first, last):calls.append((first, last));return expected[first:last]
            self.assertFalse(adapted.encode_chunks(work, folder, 'same-inputs', 13, 3, 2, 4, compute, until_chunks=1))
            first = (folder / 'chunk-0000000.npy').read_bytes()
            self.assertTrue(adapted.encode_chunks(work, folder, 'same-inputs', 13, 3, 2, 4, compute))
            np.testing.assert_array_equal(np.load(folder / 'vectors.npy'), expected)
            self.assertEqual(calls, [(0, 4), (4, 8), (8, 12), (12, 13)])
            self.assertEqual(first, (folder / 'chunk-0000000.npy').read_bytes())
            count = len(calls);adapted.encode_chunks(work, folder, 'same-inputs', 13, 3, 2, 4, compute)
            self.assertEqual(count, len(calls))
            with self.assertRaisesRegex(RuntimeError, 'Identity changed'):
                adapted.encode_chunks(work, folder, 'different-source-weights', 13, 3, 2, 4, compute)
            (folder / 'vectors.npy').write_bytes(b'corrupt')
            with self.assertRaisesRegex(RuntimeError, 'modified artifact'):
                adapted.encode_chunks(work, folder, 'same-inputs', 13, 3, 2, 4, compute)

    def test_context_excludes_current_future_and_tied_outcomes_or_vectors(self):
        rows = []
        for author in ('A', 'B'):
            for k in range(12):
                rows.append({'author': author, 'post': str(k), 'posted_at': pd.Timestamp('2020-01-01', tz='UTC') + pd.Timedelta(days=8 if k == 9 else k), 'level': 1.5})
        posts = pd.DataFrame(rows);rng = np.random.default_rng(7);vectors = rng.normal(size=(24, 4)).astype('float32')
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True);outcomes = np.arange(24, dtype=float)
        before = contextual_embeddings(posts, vectors, outcomes)
        changed = outcomes.copy();changed[8:] = 999999
        after = contextual_embeddings(posts, vectors, changed)
        np.testing.assert_array_equal(before[:10], after[:10])
        values = vectors.copy();values[9:] *= -3
        after = contextual_embeddings(posts, values, outcomes)
        np.testing.assert_array_equal(before[:9], after[:9])

    def test_complete_fold_specific_tree_oof_resume_and_test_target_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);data = root / 'data';work = root / 'work';config = make_release(data)
            config.update(enable_images=True, bootstrap_samples=3);prepare(data, work, config)
            posts = pd.read_parquet(work / 'cache/all_posts.parquet');rng = np.random.default_rng(8)
            for role, name, width in [('text_encoder', 'text_embeddings', 4), ('image_encoder', 'image_embeddings', 5)]:
                path = work / f'cache/{name}.npy';save_array(path, rng.normal(size=(len(posts), width)).astype('float32'))
                mark(work / f'cache/{name}.json', identity({'settings': config[role], 'data': digest(work / 'cache/all_posts.parquet')}), [path])
            base.train_models(work, config)
            parent_fp = identity({'code': artifacts_fingerprint(Path(base.__file__).resolve().parents[1]), 'config': config})
            atomic_json(work / 'environment/manifest.json', {'fingerprint': parent_fp, 'config': config})
            experiment = work / 'environment/refinements/r07';experiment.mkdir(parents=True);group_fp = 'synthetic-fold-specific'
            source_spec = {'name': 'synthetic-ft-source', 'max_length': 16};source_fp = 'synthetic-supervised-encoder'
            old = work / 'environment/refinements/r05';old.mkdir(parents=True)
            atomic_json(old / 'manifest.json', {'fingerprint': source_fp, 'parent_fingerprint': parent_fp,
                        'script_sha256': digest(FTHELPER), 'inputs': {n: digest(work / 'cache' / n) for n in ('train.parquet', 'test.parquet', 'prepared.json', 'image_embeddings.json')},
                        'plan': {'candidates': [source_spec]}})
            source_folder = work / 'models/synthetic-ft-source';source_folder.mkdir()
            source_candidate_fp = identity({'parent': source_fp, 'spec': source_spec})
            for job in ('fold-0', 'fold-1', 'fold-2', 'full'):
                path = source_folder / ('model.pt' if job == 'full' else job + '.pt');path.write_bytes(job.encode())
                fp = source_candidate_fp if job == 'full' else identity({'parent': source_candidate_fp, 'fold': int(job.split('-')[1])})
                outputs = [path]
                if job == 'full':
                    with np.load(work / 'models/baseline/probabilities.npz') as z:
                        np.savez_compressed(source_folder / 'probabilities.npz', **{k: z[k] for k in ('oof', 'test')})
                    metrics = json.loads((work / 'models/baseline/metrics.json').read_text());metrics['spec'] = source_spec
                    atomic_json(source_folder / 'metrics.json', metrics);outputs += [source_folder / 'probabilities.npz', source_folder / 'metrics.json']
                mark(source_folder / ('complete.json' if job == 'full' else job + '.json'), fp, outputs)
            past = posts[['author', 'post']].copy();past['rh_fixture'] = posts.level
            graph = posts[['author', 'post']].copy();graph['eg_fixture'] = posts.history_count
            past.to_parquet(experiment / 'past.parquet', index=False);graph.to_parquet(experiment / 'graph.parquet', index=False)
            atomic_json(experiment / 'graph-source.json', {'fingerprint': 'synthetic-graph'})
            trial = {'name': 'refine-r07-fixture', 'family': 'multimodal', 'parameters': {'depth': 2, 'iterations': 8, 'learning_rate': .2, 'class_power': .5}, 'pca_components': 6, 'l2_leaf_reg': 5}
            plan = {'group': 'r07', 'encoder_width': 4, 'candidates': [trial], 'max_length': 16,
                    'source_group': 'r05', 'source_group_fingerprint': source_fp, 'source_candidate': source_spec['name'],
                    'graph_manifest': str((experiment / 'graph-source.json').relative_to(work)), 'graph_group_fingerprint': 'synthetic-graph',
                    'past_features': str((experiment / 'past.parquet').relative_to(work)), 'past_features_sha256': digest(experiment / 'past.parquet'),
                    'graph_features': str((experiment / 'graph.parquet').relative_to(work)), 'graph_features_sha256': digest(experiment / 'graph.parquet')}
            plan_path = experiment / 'plan.json';atomic_json(plan_path, plan)
            _, _, _, group_fp, records = adapted.initialize(work, plan_path, HELPER, FTHELPER)
            for job, record in records.items():
                folder = experiment / 'encoders' / job;folder.mkdir(parents=True)
                values = rng.normal(size=(len(posts), 4)).astype('float32');values /= np.linalg.norm(values, axis=1, keepdims=True)
                save_array(folder / 'vectors.npy', values)
                mark(folder / 'complete.json', adapted.encoder_fp(group_fp, job, record), [folder / 'vectors.npy'])
            mark(experiment / 'encoding-complete.json', group_fp, [experiment / 'encoders' / job / 'complete.json' for job in records])
            first_frame, _, _, _ = adapted.frames(work, plan, experiment, group_fp, records, 'fold-0')
            indexes = pd.MultiIndex.from_frame(posts[['author', 'post']]).get_indexer(pd.MultiIndex.from_frame(first_frame[['author', 'post']]))
            first_vectors = np.load(experiment / 'encoders/fold-0/vectors.npy')
            full_vectors = np.load(experiment / 'encoders/full/vectors.npy')
            np.testing.assert_array_equal(first_frame.adapted_component_0.to_numpy(), first_vectors[indexes, 0])
            self.assertFalse(np.array_equal(first_frame.adapted_component_0.to_numpy(), full_vectors[indexes, 0]))
            adapted.train(work, config, plan, experiment, group_fp, records, HELPER)
            result = (work / 'predictions/final.csv').read_bytes();folder = work / 'models' / trial['name']
            self.assertEqual(len(list(folder.glob('fold-*.json'))), 3)
            for k in range(3):self.assertEqual(json.loads((folder / f'fold-{k}.json').read_text())['encoder_job'], f'fold-{k}')
            labels = pd.read_csv(data / 'labels.csv');labels.loc[labels.split == 'test', 'label'] = 2;labels.to_csv(data / 'labels.csv', index=False)
            _, _, _, resumed_fp, _ = adapted.initialize(work, plan_path, HELPER, FTHELPER)
            self.assertEqual(group_fp, resumed_fp)
            adapted.train(work, config, plan, experiment, group_fp, records, HELPER)
            self.assertEqual(result, (work / 'predictions/final.csv').read_bytes())
            self.assertFalse((work / 'report/official-test.json').exists())
            # A missing fold cache must not fall back to the available full cache.
            (experiment / 'encoders/fold-1/complete.json').unlink()
            with self.assertRaises(AssertionError):adapted.frames(work, plan, experiment, group_fp, records, 'fold-1')


if __name__ == '__main__':
    unittest.main()
