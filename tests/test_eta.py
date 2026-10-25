import importlib.util
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

path = Path(os.environ.get("KAIROS_ETA_SCRIPT", Path(__file__).resolve().parents[1] / "scripts/update-eta.py"))
spec = importlib.util.spec_from_file_location("operational_eta", path)
eta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eta)


class EtaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def put(self, name, value):
        p = self.work / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))
        return p

    def training(self):
        self.put("status.json", {"stage": "train"})
        self.put("training-status.json", {"candidate": "tfidf-1", "job": "full"})
        folder = self.work / "models/tfidf-1/checkpoints/full"
        folder.mkdir(parents=True)
        for iteration, timestamp in [(50, 1000), (100, 1100), (150, 1200)]:
            p = folder / f"optimizer-{iteration:04d}.joblib"
            p.touch(); os.utime(p, (timestamp, timestamp))

    def test_recent_checkpoint_rate_and_elapsed_time(self):
        self.training()
        result = eta.task_progress(self.work, 1220)
        self.assertEqual(result["remaining_seconds"], 480)
        self.assertAlmostEqual(result["rate_per_minute"], 30)

    def test_stale_checkpoint_has_no_false_countdown(self):
        self.training()
        self.assertFalse(eta.task_progress(self.work, 1800)["available"])

    def test_iteration_cap_does_not_claim_whole_stage_is_finished(self):
        self.training()
        p = self.work / "models/tfidf-1/checkpoints/full/optimizer-0400.joblib"
        p.touch(); os.utime(p, (1700, 1700))
        result = eta.task_progress(self.work, 1720)
        self.assertTrue(result["fit_complete"])
        self.assertIsNone(result["remaining_seconds"])
        self.assertIsNone(result["finish_at"])
        self.assertAlmostEqual(result["rate_per_minute"], 30)

    def test_resumed_text_cache_counts_prior_completed_vectors(self):
        self.put("status.json", {"stage": "text"})
        p = self.put("encoder-status.json", {"stage": "text", "new_texts": 20, "unique_texts": 100, "elapsed_seconds": 40})
        os.utime(p, (1200, 1200))
        (self.work / "cache").mkdir()
        with sqlite3.connect(self.work / "cache/text.sqlite") as db:
            db.execute("CREATE TABLE vectors (key TEXT PRIMARY KEY, data BLOB)")
            db.executemany("INSERT INTO vectors VALUES (?,?)", [(str(i), b'') for i in range(80)])
        result = eta.task_progress(self.work, 1200)
        self.assertEqual(result["done"], 80)
        self.assertEqual(result["remaining_seconds"], 40)

    def test_stop_and_previous_stage_progress_are_not_active_eta(self):
        self.put("status.json", {"stage": "images"})
        self.put("encoder-status.json", {"stage": "text", "new_texts": 20})
        self.assertFalse(eta.task_progress(self.work, 1200)["available"])
        (self.work / "STOP").touch()
        self.assertFalse(eta.task_progress(self.work, 1200)["available"])

    def test_new_refinement_does_not_reuse_previous_group_fit(self):
        self.training()
        self.put("status.json", {"stage": "train", "extension": "r02"})
        self.put("training-status.json", {"candidate": "refine-r01-rich-d9-p256", "stage": "full_fit"})
        self.put("models/refine-r01-rich-d9-p256/checkpoints/full/catboost_training.json",
                 {"meta": {"iteration_count": 2800},
                  "iterations": [{"iteration": 1000, "passed_time": 200}]})
        result = eta.task_progress(self.work, 1200)
        self.assertFalse(result["available"])
        self.assertIn("r02", result["task"])
        self.assertNotIn("rate_per_minute", result)

    def test_download_eta_uses_expected_sizes_and_partial_bytes(self):
        self.put("status.json", {"stage": "fetch-models"})
        self.put("environment/eta-model-files.json", {"files": {"weights.bin": 100}})
        self.put("environment/runtime-fetch-models.json", {"at": "1970-01-01T00:16:40+00:00"})
        models = self.work / "weights"
        models.mkdir()
        (models / "weights.bin.partial").write_bytes(b'x' * 50)
        result = eta.task_progress(self.work, 1010, models)
        self.assertEqual(result["remaining_seconds"], 10)
        self.assertEqual(result["done"], 50)

    def test_neural_eta_uses_current_session_after_resume(self):
        self.put("status.json", {"stage": "train", "extension": "r03"})
        self.put("training-status.json", {"candidate": "refine-r03-mlp", "job": "fold-0", "optimizer": "pytorch", "at": "2026-01-01T00:00:00+00:00"})
        p = self.put("neural-progress.json", {"candidate": "refine-r03-mlp", "job": "fold-0", "iterations": 600,
                     "total_iterations": 1000, "session_iterations": 100, "session_elapsed_seconds": 20,
                     "at": "2026-01-01T00:01:00+00:00"})
        os.utime(p, (1200, 1200))
        result = eta.task_progress(self.work, 1200)
        self.assertEqual(result["remaining_seconds"], 80)
        self.assertAlmostEqual(result["rate_per_minute"], 300)
        self.assertFalse(eta.task_progress(self.work, 1800)["available"])
        self.put("neural-progress.json", {"candidate": "refine-r03-mlp", "job": "fold-1"})
        self.assertFalse(eta.task_progress(self.work, 1200)["available"])

    def test_neural_eta_rejects_previous_attempt_progress(self):
        self.put("status.json", {"stage": "train", "extension": "r03"})
        self.put("training-status.json", {"candidate": "refine-r03-mlp", "job": "full", "optimizer": "pytorch", "at": "2026-01-01T01:00:00+00:00"})
        self.put("neural-progress.json", {"candidate": "refine-r03-mlp", "job": "full", "iterations": 600,
                 "total_iterations": 1000, "session_iterations": 600, "session_elapsed_seconds": 20,
                 "at": "2026-01-01T00:01:00+00:00"})
        self.assertFalse(eta.task_progress(self.work, 1200)["available"])


if __name__ == "__main__":
    unittest.main()
