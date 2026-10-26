import unittest
import numpy as np
import pandas as pd
from scripts.empty_text_models import route_predictions,guard_context,choose_route,balanced_decision,BASE_NAMES,OCR_NAMES
from scripts.ocr_empty_posts import clean_fragment


class EmptyTextTests(unittest.TestCase):
    def test_route_cannot_change_posts_with_original_text(self):
        base = np.array([0,1,2,1])
        actual = route_predictions(base,np.array([2,0]),[True,False,False,True])
        np.testing.assert_array_equal(actual,[2,1,2,0])
        np.testing.assert_array_equal(base,[0,1,2,1])
        with self.assertRaises(ValueError):
            route_predictions(base,[2],[True,False,False,True])

    def test_guard_masks_only_context_comparisons_for_empty_originals(self):
        table = pd.DataFrame(dict(original_chars=[0,10],nov_expanded_max_100=[.9,.8],
            nbr_expanded_top1_delta=[1.,2.],nov_expanded_past_available=[5,8],nov_own_max_100=[.5,.4]))
        result,masked = guard_context(table,list(table.columns))
        self.assertEqual(set(masked),{"nov_expanded_max_100","nbr_expanded_top1_delta"})
        self.assertTrue(result.loc[0,masked].isna().all())
        pd.testing.assert_series_equal(result.loc[1],table.loc[1])
        np.testing.assert_array_equal(result.nov_expanded_past_available,[5,8])
        np.testing.assert_array_equal(result.nov_own_max_100,[.5,.4])

    def test_route_selection_keeps_noop_or_chooses_inner_improvement(self):
        labels = np.array([0,1,2,0,1,2])
        mask = np.array([True,True,True,False,False,False])
        table = pd.DataFrame({"current":labels})
        for name in BASE_NAMES+OCR_NAMES:
            table[name] = [1,1,1,0,1,2]
        self.assertEqual(choose_route(labels,mask,table)["chosen"],"current")
        table.current = [0,1,0,0,1,2]
        table.old_visual_route = labels
        self.assertEqual(choose_route(labels,mask,table)["chosen"],"old_visual_route")

    def test_ocr_removes_counter_fragments_and_contact_tokens(self):
        self.assertEqual(clean_fragment("123 likes"),"")
        self.assertEqual(clean_fragment("Комментарии 45"),"")
        self.assertEqual(clean_fragment("New methods 2026 in science"),"New methods in science")
        self.assertEqual(clean_fragment("Contact maria@example.com https://example.org @someone"),"Contact")
        self.assertEqual(clean_fragment("Проверка данных"),"Проверка данных")
        self.assertEqual(clean_fragment("123"),"")

    def test_balanced_selection_retains_perfect_baseline(self):
        labels = np.array([0,1,2,0,1,2])
        mask = np.array([True,True,True,False,False,False])
        config = balanced_decision(labels,mask,labels,{"current":np.eye(3)},dict(multipliers=[1.,1.,1.]))
        self.assertEqual(config["chosen"],"current")
        self.assertEqual(config["multipliers"],[1.,1.,1.])
        self.assertEqual(config["calibration_overall"],1.)


if __name__ == "__main__":
    unittest.main()
