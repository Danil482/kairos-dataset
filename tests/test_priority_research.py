import unittest
import numpy as np
import pandas as pd
from scipy.optimize import check_grad
from scripts.priority_features import contextual_history, content_fields
from scripts.priority_models import ordinal_probabilities, gate_objective, split_gate_authors


class PriorityResearchTests(unittest.TestCase):
    def sample(self):
        posts = pd.DataFrame(dict(author=["a"]*5, post=list("abcde"),
            posted_at=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-10", "2024-01-20"], utc=True),
            reactions=[1, 100, 1, 3, 20], content_type=["text", "image", "image", "text", "text"], quote_mode=[0, 1, 2, 0, 1]))
        vectors = np.array([[1.,0.], [0.,1.], [1.,0.], [1.,0.], [0.,1.]])
        return posts, vectors

    def test_history_excludes_current_future_and_timestamp_ties(self):
        posts, vectors = self.sample()
        for mode in ("type", "quote", "recency"):
            expected = contextual_history(posts, vectors, mode, "x", [-.5, 1.])
            for i in range(len(posts)):
                changed = posts.copy()
                future = changed.posted_at.ge(changed.posted_at.iloc[i]).to_numpy(copy=True)
                changed.loc[future, "reactions"] = 1000000
                future[i] = False
                altered = vectors.copy()
                altered[future] = [-1., 0.]
                actual = contextual_history(changed, altered, mode, "x", [-.5, 1.])
                pd.testing.assert_series_equal(expected.iloc[i], actual.iloc[i])
        result = contextual_history(posts, vectors, "type", "x", [-.5, 1.])
        self.assertEqual(result.x_n.iloc[1], 0)
        self.assertEqual(result.x_n.iloc[2], 0)
        self.assertEqual(result.x_n.iloc[3], 1)

    def test_author_isolation_row_order_and_time_units(self):
        posts, vectors = self.sample()
        expected = contextual_history(posts, vectors, "quote", "x", [-.5, 1.])
        mixed = pd.concat([posts, posts.assign(author="b", reactions=99999)], ignore_index=True)
        actual = contextual_history(mixed, np.vstack([vectors, -vectors]), "quote", "x", [-.5, 1.])
        pd.testing.assert_frame_equal(expected, actual.iloc[:len(posts)])
        order = np.array([4, 2, 0, 3, 1])
        changed = posts.iloc[order].copy()
        changed["posted_at"] = changed.posted_at.dt.as_unit("ms")
        actual = contextual_history(changed, vectors[order], "quote", "x", [-.5, 1.])
        pd.testing.assert_frame_equal(expected, actual.iloc[np.argsort(order)].reset_index(drop=True))

    def test_small_context_is_shrunk_and_empty_vectors_not_neighbors(self):
        posts, vectors = self.sample()
        posts.loc[4, "quote_mode"] = 1
        result = contextual_history(posts, vectors, "quote", "x", [-.5, 1.])
        self.assertEqual(result.x_n.iloc[4], 1)
        raw_delta = np.log1p(100)-np.median(np.log1p([1,100,1,3]))
        self.assertGreater(result.x_mean_delta.iloc[4], 0)
        self.assertLess(result.x_mean_delta.iloc[4], raw_delta/5)
        vectors[:] = 0
        result = contextual_history(posts, vectors, "quote", "x", [-.5, 1.])
        self.assertTrue(result.x_centroid.isna().all())
        self.assertTrue(result.x_neighbor_delta.eq(0).all())

    def test_fields_use_content_allowlist_and_explicit_quote_mode(self):
        post = dict(text="Comment", totalReactionCount=999, author=dict(name="PRIVATE"),
                    resharedPost=dict(text="Quoted text", author=dict(name="PRIVATE"),
                                      document=dict(title="Document", transcript="Body")))
        result = content_fields(post)
        self.assertEqual(result["quote_mode"], 1)
        self.assertNotIn("PRIVATE", str(result))
        self.assertNotIn("999", str(result))
        self.assertIn("Body", result["quote_text"])
        post["text"] = ""
        self.assertEqual(content_fields(post)["quote_mode"], 2)
        post["resharedPost"] = None
        self.assertEqual(content_fields(post)["quote_mode"], 0)

    def test_ordinal_projection_is_valid_for_crossings_and_extremes(self):
        values = ordinal_probabilities([.8, .1, 0., 1.], [.3, .9, 0., 1.])
        np.testing.assert_allclose(values, [[.2,.5,.3], [.5,0.,.5], [1.,0.,0.], [0.,0.,1.]])
        self.assertTrue((values>=0).all())
        np.testing.assert_allclose(values.sum(axis=1), 1.)
        with self.assertRaises(ValueError):
            ordinal_probabilities([1.1], [0.])

    def test_gate_gradient_and_author_separation(self):
        rng = np.random.default_rng(14)
        x = np.column_stack([np.ones(30), rng.normal(size=(30, 4))])
        a, b = rng.dirichlet([2,2,2], 30), rng.dirichlet([2,2,2], 30)
        labels, coef = rng.integers(0,3,30), rng.normal(size=5)
        args = (x,a,b,labels,.1)
        error = check_grad(lambda z:gate_objective(z,*args)[0], lambda z:gate_objective(z,*args)[1], coef)
        self.assertLess(error, 1e-6)
        table = pd.DataFrame(dict(author=np.repeat(list("abcdefgh"), 3)))
        mask = split_gate_authors(table)
        self.assertFalse(set(table.author[mask]) & set(table.author[~mask]))
        np.testing.assert_array_equal(mask, split_gate_authors(table))


if __name__ == "__main__":
    unittest.main()
