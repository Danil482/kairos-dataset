import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(os.environ.get("KAIROS_GPU_SELECTOR_SCRIPT", Path(__file__).resolve().parents[1] / "scripts/select-free-gpu.sh"))
GPU0 = "GPU-00000000-0000-0000-0000-000000000000"
GPU3 = "GPU-33333333-3333-3333-3333-333333333333"
GPU4 = "GPU-44444444-4444-4444-4444-444444444444"


class GpuSelectionTests(unittest.TestCase):
    def select(self, rows, processes=None, failure=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "nvidia-smi"
            fake.write_text(f"#!{sys.executable}\nimport json,sys\nrows={rows!r}\nprocesses={processes or {}!r}\nif '-i' in sys.argv:\n print(processes.get(sys.argv[sys.argv.index('-i')+1],''),end='')\nelse:\n sys.stdout.write(rows)\n sys.exit({69 if failure else 0})\n")
            fake.chmod(0o700)
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'])
            return subprocess.run(['bash', str(SCRIPT), '18000', GPU3], env=env, capture_output=True, text=True)

    def test_busy_preferred_falls_back_to_other_idle_card(self):
        rows = f"3, {GPU3}, L40S, 45000, 0, 0\n4, {GPU4}, L40, 45000, 0, 0\n"
        result = self.select(rows, {GPU3:'12345\n'})
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.split('\t')[0], GPU4)

    def test_idle_preferred_is_kept_even_with_another_free_card(self):
        rows = f"0, {GPU0}, H100, 80000, 0, 0\n3, {GPU3}, L40S, 45000, 0, 0\n"
        self.assertEqual(self.select(rows).stdout.split('\t')[0], GPU3)

    def test_unused_memory_is_not_enough_if_card_is_occupied(self):
        rows = f"0, {GPU0}, H100, 79000, 1000, 0\n4, {GPU4}, L40, 45000, 0, 0\n"
        self.assertEqual(self.select(rows).stdout.split('\t')[0], GPU4)

    def test_all_busy_returns_waiting_code_without_device(self):
        result = self.select(f"0, {GPU0}, H100, 79000, 1000, 80\n")
        self.assertEqual(result.returncode, 75)
        self.assertEqual(result.stdout, '')

    def test_insufficient_memory_and_nvml_failure_fail_closed(self):
        self.assertEqual(self.select(f"4, {GPU4}, small, 8000, 0, 0\n").returncode, 75)
        self.assertEqual(self.select('', failure=True).returncode, 69)


if __name__ == '__main__':
    unittest.main()
