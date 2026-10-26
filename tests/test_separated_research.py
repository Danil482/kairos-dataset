import unittest
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scripts.separated_models import split_stopping_decision, residual_vectors, extend_projection, choose_variant, VARIANTS


class SeparatedTests(unittest.TestCase):
    def test_roles_preserve_authors_and_are_order_label_invariant(self):
        frame = pd.DataFrame(dict(author=np.repeat(np.arange(11),3),label=np.arange(33)%3))
        stop,decision = split_stopping_decision(frame)
        self.assertFalse(set(stop.author)&set(decision.author))
        self.assertEqual(set(stop.index)|set(decision.index),set(frame.index))
        changed = frame.sample(frac=1,random_state=4).assign(label=0)
        other_stop,other_decision = split_stopping_decision(changed)
        self.assertEqual(set(stop.author),set(other_stop.author))
        self.assertEqual(set(decision.author),set(other_decision.author))
        with self.assertRaises(ValueError):
            split_stopping_decision(frame.iloc[:3])

    def test_nested_projection_preserves_original_information(self):
        rng = np.random.default_rng(42)
        fitting = rng.normal(size=(340,280))
        heldout = rng.normal(size=(20,280))
        old = PCA(n_components=64,svd_solver='full').fit(fitting)
        residual = PCA(n_components=192,svd_solver='full').fit(residual_vectors(old,fitting))
        self.assertLess(np.abs(old.components_ @ residual.components_.T).max(),1e-8)
        original = old.transform(heldout)
        extra = residual.transform(residual_vectors(old,heldout))
        p64,p128,p256 = [extend_projection(original,extra,n) for n in (64,128,256)]
        np.testing.assert_array_equal(p64,p128[:,:64])
        np.testing.assert_array_equal(p128,p256[:,:128])
        errors = []
        for n in (0,64,192):
            reconstruction = old.inverse_transform(original)+residual.mean_+extra[:,:n] @ residual.components_[:n]
            errors.append(np.square(heldout-reconstruction).sum())
        self.assertGreater(errors[0],errors[1])
        self.assertGreater(errors[1],errors[2])

    def test_selector_uses_fixed_order_for_ties_and_rejects_missing_scores(self):
        scores = dict.fromkeys(VARIANTS,.5)
        self.assertEqual(choose_variant(scores),'base64')
        scores['fields64'] = .6
        self.assertEqual(choose_variant(scores),'fields64')
        with self.assertRaises(ValueError):
            choose_variant({'base64':.9})
        scores['e5_128'] = np.nan
        with self.assertRaises(ValueError):
            choose_variant(scores)


if __name__ == '__main__':
    unittest.main()
