import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from scripts.stability_models import (author_confusions, macro_from_confusion, stable_decisions,
                                      bootstrap_difference, FINE_GRID)
from scripts.calibrate_lecture_decisions import select_decision, GRID
from train_maria import bootstrap_authors


class StabilityTests(unittest.TestCase):
    def sample(self):
        rng = np.random.default_rng(8)
        labels = rng.integers(0,3,180)
        probabilities = rng.dirichlet([1,2,1],180)
        authors = np.repeat(np.arange(30),6)
        return labels,probabilities,authors

    def test_author_confusions_reproduce_pooled_metric(self):
        y,p,a = self.sample()
        pred = p.argmax(axis=1)
        matrix = author_confusions(y,pred,a)
        np.testing.assert_array_equal(matrix.sum(axis=0),confusion_matrix(y,pred,labels=[0,1,2]))
        self.assertAlmostEqual(float(macro_from_confusion(matrix.sum(axis=0))),f1_score(y,pred,average="macro"))
        # Repeating one author gives all of that author's posts the same multiplicity.
        expected = confusion_matrix(np.r_[y,y[a==0]],np.r_[pred,pred[a==0]],labels=[0,1,2])
        np.testing.assert_array_equal(matrix.sum(axis=0)+matrix[0],expected)

    def test_grid_matches_existing_selector_and_fine_grid_contains_old_grid(self):
        y,p,a = self.sample()
        coarse,_ = stable_decisions(y,p,a,GRID,repeats=12)
        self.assertEqual(coarse,select_decision(y,p))
        self.assertEqual(len(FINE_GRID),23)
        self.assertTrue(set(GRID)<=set(FINE_GRID))

    def test_bootstrap_decision_is_deterministic_and_order_invariant(self):
        y,p,a = self.sample()
        _,expected = stable_decisions(y,p,a,[.5,1.,2.],repeats=20)
        order = np.random.default_rng(3).permutation(len(y))
        _,actual = stable_decisions(y[order],p[order],a[order],[.5,1.,2.],repeats=20)
        self.assertEqual(expected,actual)
        self.assertEqual(actual["multipliers"][1],1.)
        self.assertTrue(np.all(np.asarray(actual["multipliers"])>0))

    def test_paired_interval_is_zero_for_same_predictions_and_ignores_author_names(self):
        y,p,a = self.sample()
        pred = p.argmax(axis=1)
        interval = bootstrap_difference(y,pred,pred,a,repeats=30)
        self.assertEqual(interval["low"],0.)
        self.assertEqual(interval["high"],0.)
        # Equality must also hold with string identifiers and unequal author sizes.
        interval = bootstrap_difference(y[:137],pred[:137],pred[:137],a[:137].astype(str),repeats=30)
        self.assertEqual(interval["high"],0.)

    def test_bad_calibration_arrays_are_rejected(self):
        with self.assertRaises(ValueError):
            stable_decisions([0],[[1.,1.,1.]],["a"])
        with self.assertRaises(ValueError):
            stable_decisions([0],[[1.,0.,0.]],[])
        with self.assertRaises(ValueError):
            author_confusions([3],[0],["a"])

    def test_fast_paired_interval_matches_existing_author_bootstrap(self):
        y,p,a = self.sample()
        first = p.argmax(axis=1)
        second = (p*np.array([2.,1.,1.5])).argmax(axis=1)
        table = pd.DataFrame(dict(author=a,label=y,first=first,second=second))
        expected = bootstrap_authors(table,"first","second",repeats=70)
        actual = bootstrap_difference(y,first,second,a,repeats=70)
        self.assertEqual(actual,expected)


if __name__ == "__main__":
    unittest.main()
