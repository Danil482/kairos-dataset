import unittest
import numpy as np
from scripts.evaluate_frozen_test import blend_folds

class FrozenInferenceTests(unittest.TestCase):
    def test_fold_rules_normalized_before_equal_combination(self):
        first=np.array([[.1,.6,.3],[0,.4,.6]])
        second=np.array([[.4,.5,.1],[0,.7,.3]])
        a=first*np.array([2,1,3]);a/=a.sum(axis=1,keepdims=True)
        b=second*np.array([.5,1,2]);b/=b.sum(axis=1,keepdims=True)
        actual=blend_folds([first,second],[[2,1,3],[.5,1,2]])
        np.testing.assert_allclose(actual,(a+b)/2)
        np.testing.assert_allclose(actual.sum(axis=1),1)
        self.assertEqual(actual[1,0],0)

    def test_order_invariance_and_reject_invalid_rules(self):
        a=np.array([[.2,.5,.3]])
        b=np.array([[.5,.4,.1]])
        np.testing.assert_allclose(blend_folds([a,b],[[1,1,2],[2,1,1]]),blend_folds([b,a],[[2,1,1],[1,1,2]]))
        with self.assertRaises(ValueError):blend_folds([a],[[0,1,1]])
        with self.assertRaises(ValueError):blend_folds([a,b],[[1,1,1]])
        with self.assertRaises(ValueError):blend_folds([],[])

if __name__=='__main__':unittest.main()
