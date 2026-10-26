"""Semantic history must never see the current outcome, future, or other authors."""
import unittest
import numpy as np
import pandas as pd
from scripts.semantic_history import history_features


class SemanticHistoryTests(unittest.TestCase):
    def sample(self):
        table = pd.DataFrame(dict(author=["a"]*5,post=list("abcde"),
            posted_at=pd.to_datetime(["2024-01-01","2024-01-02","2024-01-02","2024-01-10","2024-02-01"],utc=True),
            reactions=[0,2,10,20,50],content_type=["text"]*5))
        vectors = np.array([[1.,0.],[0.,1.],[1.,0.],[1.,0.],[0.,1.]],dtype=np.float32)
        return table,vectors

    def calculate(self,table,vectors):
        return history_features(table,vectors,"test",[-.55,1.15])

    def test_current_and_future_reactions_cannot_change_current_features(self):
        table,vectors = self.sample()
        expected = self.calculate(table,vectors)
        for i in range(len(table)):
            changed = table.copy()
            changed.loc[changed.posted_at.ge(table.posted_at.iloc[i]),"reactions"] = 1000000
            actual = self.calculate(changed,vectors)
            pd.testing.assert_series_equal(expected.iloc[i],actual.iloc[i])

    def test_future_and_simultaneous_texts_are_not_history(self):
        table,vectors = self.sample()
        expected = self.calculate(table,vectors)
        for i in range(len(table)):
            changed = vectors.copy()
            mask = table.posted_at.ge(table.posted_at.iloc[i]).to_numpy(copy=True)
            mask[i] = False
            changed[mask] = [-1.,0.]
            actual = self.calculate(table,changed)
            pd.testing.assert_series_equal(expected.iloc[i],actual.iloc[i])
        self.assertEqual(expected.nov_test_past_available.iloc[1],1)
        self.assertEqual(expected.nov_test_past_available.iloc[2],1)
        self.assertEqual(expected.nov_test_last_similarity.iloc[1],0)
        self.assertEqual(expected.nov_test_last_similarity.iloc[2],1)

    def test_author_isolation_and_unlabelled_warmup(self):
        table,vectors = self.sample()
        expected = self.calculate(table,vectors)
        mixed = pd.concat([table,table.assign(author="b",reactions=9999)],ignore_index=True)
        actual = self.calculate(mixed,np.vstack([vectors,-vectors]))
        pd.testing.assert_frame_equal(expected,actual.iloc[:len(table)])
        # The first query includes all three earlier posts even if they have no label.
        self.assertEqual(expected.nov_test_past_available.iloc[3],3)
        self.assertAlmostEqual(expected.nov_test_centroid_all.iloc[3],2/np.sqrt(5))
        self.assertEqual(expected.nov_test_novelty_100.iloc[3],0)

    def test_missing_text_is_preserved_and_does_not_form_a_neighbor(self):
        table,vectors = self.sample()
        vectors[0] = 0
        vectors[3] = 0
        actual = self.calculate(table,vectors)
        self.assertEqual(len(actual),len(table))
        self.assertEqual(actual.nov_test_past_available.iloc[1],0)
        self.assertTrue(np.isnan(actual.nov_test_centroid_all.iloc[1]))
        self.assertEqual(actual.nov_test_current_available.iloc[3],0)
        self.assertTrue(np.isnan(actual.nbr_test_top1_delta.iloc[3]))
        self.assertEqual(actual.nov_test_past_available.iloc[4],2)

    def test_timestamp_units_and_input_order_do_not_change_features(self):
        table,vectors = self.sample()
        expected = self.calculate(table,vectors)
        for unit in ("ms","us","ns"):
            changed = table.copy()
            changed["posted_at"] = changed.posted_at.dt.as_unit(unit)
            pd.testing.assert_frame_equal(expected,self.calculate(changed,vectors))
        order = np.array([4,2,0,3,1])
        actual = self.calculate(table.iloc[order],vectors[order])
        pd.testing.assert_frame_equal(expected,actual.iloc[np.argsort(order)].reset_index(drop=True))


if __name__ == "__main__":
    unittest.main()
