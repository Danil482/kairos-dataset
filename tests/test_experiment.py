"""Author-held-out end-to-end checks on a synthetic release, not benchmark scores."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from test_text_pipeline import make_release
from kairos import experiment as exp
from kairos.experiment_features import load_experiment_data


class ExperimentTest(unittest.TestCase):
    def test_validate_fold_refit_and_exact_test_keys(self):
        with tempfile.TemporaryDirectory(prefix="sma-experiment-test-") as tmp:
            root = Path(tmp)
            release = make_release(root / "data")
            table, history, groups, data_report = load_experiment_data(release)
            config = json.loads((exp.ROOT / "configs/cpu.json").read_text())
            config["hgb"]["max_iter"] = 5
            config["text"].update(min_df=1, max_features=500, svd_components=8, svd_n_iter=2)
            config["cv_folds"] = 3
            output = root / "output"
            output.mkdir()
            manifest = exp.fingerprint(config, data_report)
            with threadpool_limits(2):
                report = exp.validate(table, history, groups, config, output, manifest)
                self.assertFalse(set(report["split"]["fit_authors"]) &
                                 set(report["split"]["validation_authors"]))
                self.assertEqual(report["split"]["n_fit"] + report["split"]["n_validation"], 480)
                self.assertIn("E3e_svd_neighbors", report["recipes"])
                cv = exp.stability(table, history, config, report, output)
                fold_authors = [set(v) for v in cv["fold_authors"].values()]
                self.assertEqual(len(set.union(*fold_authors)), 24)
                self.assertTrue(all(not a & b for i, a in enumerate(fold_authors)
                                    for b in fold_authors[i+1:]))
                # Explicitly exercise the learned SVD/history final-refit path.
                exp.final(table, history, config, report, "E3e_svd_neighbors", output)
            predictions = pd.read_csv(output / "final/predictions.csv")
            expected = table.loc[table.split == "test", exp.KEYS].reset_index(drop=True)
            pd.testing.assert_frame_equal(predictions[exp.KEYS], expected)
            self.assertTrue(predictions.label.isin([0, 1, 2]).all())
            self.assertTrue(table.loc[table.split == "test", "label"].isna().all())
            self.assertFalse(json.loads((output / "final/manifest.json").read_text())[
                "test_labels_used_for_fitting_or_selection"])
            self.assertTrue((output / "final/model.joblib").is_file())

    def test_text_transform_fit_excludes_validation_only_words(self):
        with tempfile.TemporaryDirectory(prefix="sma-fit-vocabulary-") as tmp:
            table, history, _, _ = load_experiment_data(make_release(Path(tmp) / "data"))
            config = json.loads((exp.ROOT / "configs/cpu.json").read_text())
            config["text"].update(min_df=1, svd_components=8, svd_n_iter=2)
            fit, val = exp.partition(table, config)
            val_authors = set(table.iloc[val].author)
            table.loc[table.author.isin(val_authors), "content_text"] += " heldoutuniquetoken"
            history.loc[history.author.isin(val_authors), "content_text"] += " heldoutuniquetoken"
            context = exp.FeatureContext(table, history, fit, config)
            with threadpool_limits(2):
                context.vector_values("svd")
            self.assertNotIn("heldoutuniquetoken", context.svd.named_steps["tfidf"].vocabulary_)


if __name__ == "__main__":
    unittest.main()
