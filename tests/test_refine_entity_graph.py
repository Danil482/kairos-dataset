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
from tests.fixtures import make_release

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(os.environ.get('KAIROS_GRAPH_SCRIPT', ROOT / 'scripts/refine-entity-graph.py'))
HELPER = Path(os.environ.get('KAIROS_GRAPH_HELPER', ROOT / 'scripts/refine-native-text.py'))
spec = importlib.util.spec_from_file_location('graph_extension', SCRIPT)
graph = importlib.util.module_from_spec(spec);spec.loader.exec_module(graph)
spec = importlib.util.spec_from_file_location('graph_native_helper_test', HELPER)
helper = importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


def toy_frames():
    rows, outcomes, raw = [], [], []
    for author in ('A', 'B'):
        for j in range(12):
            day = 8 if j == 9 else j
            key = {'author': author, 'post': f'{author}{j}'}
            rows.append({**key, 'posted_at': pd.Timestamp('2020-01-01', tz='UTC') + pd.Timedelta(days=day),
                         'level': 1.5, 'history_count': min(j, day)})
            outcomes.append({**key, 'reactions': j * j + 1})
            raw.append({**key, 'raw_text': 'text ⟨ORG_7⟩ ⟨PERSON_1⟩ ⟨AUTHOR_3⟩'})
    return pd.DataFrame(rows), pd.DataFrame(outcomes), pd.DataFrame(raw)


class EntityGraphTests(unittest.TestCase):
    def test_outcomes_are_own_author_past_only_and_tied_dates_excluded(self):
        posts, outcomes, raw = toy_frames()
        before = graph.graph_features(posts, outcomes, raw, [-.55, 1.15])
        changed = outcomes.copy();changed.loc[8:, 'reactions'] = 9999999
        after = graph.graph_features(posts, changed, raw, [-.55, 1.15])
        pd.testing.assert_frame_equal(before.iloc[:10], after.iloc[:10])
        changed = outcomes.copy();changed.loc[:7, 'reactions'] = 9999999
        affected = graph.graph_features(posts, changed, raw, [-.55, 1.15])
        self.assertNotEqual(before.loc[8, 'eg_residual_mean5'], affected.loc[8, 'eg_residual_mean5'])
        self.assertEqual(before.loc[8, 'eg_global_mentions_mean'], affected.loc[8, 'eg_global_mentions_mean'])

    def test_global_graph_uses_only_strictly_earlier_unlabeled_structure(self):
        posts, outcomes, raw = toy_frames()
        before = graph.graph_features(posts, outcomes, raw, [-.55, 1.15])
        changed = raw.copy()
        # A9 and B8/B9 tie with the A8 query; all other edited posts are later.
        changed.loc[(posts.posted_at >= posts.loc[8, 'posted_at']) & (posts.index != 8), 'raw_text'] = 'future ⟨ORG_99⟩'
        after = graph.graph_features(posts, outcomes, changed, [-.55, 1.15])
        pd.testing.assert_frame_equal(before.iloc[:9], after.iloc[:9])
        changed = raw.copy();changed.loc[12:19, 'raw_text'] = 'past other author ⟨ORG_99⟩'
        affected = graph.graph_features(posts, outcomes, changed, [-.55, 1.15])
        self.assertNotEqual(before.loc[8, 'eg_global_mentions_mean'], affected.loc[8, 'eg_global_mentions_mean'])
        own = [c for c in before if c.startswith(('eg_residual_', 'eg_class', 'eg_own_'))]
        pd.testing.assert_series_equal(before.loc[8, own], affected.loc[8, own])

    def test_raw_allowlist_and_person_scope_preserve_actual_global_ids(self):
        a = graph.entities('⟨PERSON_1⟩ ⟨ORG_007⟩', 'A')
        b = graph.entities('⟨PERSON_1⟩ ⟨ORG_7⟩', 'B')
        self.assertEqual(a & b, {('ORG', '7', '')})
        text = graph.native_raw_text(pd.Series(['hello ⟨ORG_123⟩ ⟨AUTHOR_45⟩ ⟨PERSON_67⟩']), helper)[0]
        self.assertIn('entityorg123', text);self.assertIn('entityauthor45', text)
        self.assertNotIn('67', text)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);(root / 'posts').mkdir()
            path = root / 'posts/AUTHOR_0001.json'
            feed = {'author': 'AUTHOR_0001', 'posts': [
                {'urn': 'p1', 'author': {'code': 'AUTHOR_0001'}, 'text': '⟨ORG_123⟩', 'totalReactionCount': 10},
                {'urn': 'p2', 'author': {'code': 'OTHER'}, 'text': 'repost'}]}
            path.write_text(json.dumps(feed));first, _ = graph.load_raw_posts(root)
            feed['posts'][0]['totalReactionCount'] = 999999
            feed['posts'][0]['comments'] = [{'private': 'ignored'}]
            path.write_text(json.dumps(feed));second, _ = graph.load_raw_posts(root)
            pd.testing.assert_frame_equal(first, second)
            self.assertEqual(first.columns.tolist(), ['author', 'post', 'raw_text'])
            self.assertEqual(len(first), 1)

    def test_complete_native_graph_oof_resume_and_test_target_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);data = root / 'data';work = root / 'work'
            config = make_release(data);config.update(enable_images=True, bootstrap_samples=3)
            for path in (data / 'posts').glob('*.json'):
                feed = json.loads(path.read_text())
                for j, post in enumerate(feed['posts']):
                    post['text'] += f' ⟨ORG_{j % 3}⟩ ⟨AUTHOR_{j % 4}⟩ ⟨PERSON_{j % 5}⟩'
                path.write_text(json.dumps(feed))
            prepare(data, work, config)
            posts = pd.read_parquet(work / 'cache/all_posts.parquet')
            self.assertFalse(posts.text.str.contains(r'⟨ORG_\d+⟩').any())
            raw, _ = graph.load_raw_posts(data)
            self.assertTrue(raw.raw_text.str.contains(r'⟨ORG_\d+⟩').any())
            rng = np.random.default_rng(7)
            for role, filename, width in [('text_encoder', 'text_embeddings', 4), ('image_encoder', 'image_embeddings', 5)]:
                np.save(work / f'cache/{filename}.npy', rng.normal(size=(len(posts), width)).astype('float32'))
                marker_fp = identity({'settings': config[role], 'data': digest(work / 'cache/all_posts.parquet')})
                mark(work / f'cache/{filename}.json', marker_fp, [work / f'cache/{filename}.npy'])
            base.train_models(work, config)
            parent = identity({'code': artifacts_fingerprint(Path(base.__file__).resolve().parents[1]), 'config': config})
            atomic_json(work / 'environment/manifest.json', {'fingerprint': parent, 'config': config})
            preview = root / 'preview';preview.mkdir()
            *_, past_sha = helper.build_frames(work, config, preview)
            plan = {'past_features_sha256': past_sha, 'candidates': [{
                'name': 'refine-r90-raw-graph', 'family': 'multimodal', 'parameters': {
                    'depth': 2, 'iterations': 8, 'learning_rate': .2, 'class_power': .5},
                'pca_components': 12, 'l2_leaf_reg': 5, 'rich_history': True,
                'embedding_mode': 'current-text', 'native_text': True,
                'text_min_count': 1, 'text_dictionary_size': 100, 'bow_features': 100}]}
            plan_path = root / 'plan.json';plan_path.write_text(json.dumps(plan))
            result = graph.run(work, data, plan_path, 'r90', HELPER)
            first = (work / 'predictions/final.csv').read_bytes()
            markers = {str(p): p.read_bytes() for p in (work / 'models/refine-r90-raw-graph').glob('fold-*.json')}
            labels = pd.read_csv(data / 'labels.csv');labels.loc[labels.split == 'test', 'label'] = 2
            labels.to_csv(data / 'labels.csv', index=False)
            graph.run(work, data, plan_path, 'r90', HELPER)
            self.assertEqual(first, (work / 'predictions/final.csv').read_bytes())
            self.assertTrue(all(Path(p).read_bytes() == value for p, value in markers.items()))
            self.assertEqual(len(markers), 3)
            self.assertGreater(json.loads((work / 'environment/refinements/r90/feature-audit.json').read_text())['raw_entity_posts'], 0)
            self.assertFalse((work / 'report/official-test.json').exists())
            self.assertEqual(result['partition'], 'train_author_oof')


if __name__ == '__main__':
    unittest.main()
