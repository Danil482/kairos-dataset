import hashlib
import http.server
import os
import tempfile
import threading
import time
import sys
import unittest
from pathlib import Path

from kairos_lab.acquire import archive_entries, download_file, sevenzip
from kairos_lab.common import Stopped, run_command


class OperationTests(unittest.TestCase):
    def test_large_archive_listing_is_streamed_without_repeated_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "7z"
            fake.write_text("#!/usr/bin/env python3\nimport getpass,sys\ngetpass.getpass('Enter password: ')\nsys.stdout.write('----------\\n')\nfor i in range(50000):\n sys.stdout.write(f'Path = media/{i:064d}.jpg\\nSize = 10\\nAttributes = A\\n\\n')\n")
            fake.chmod(0o700)
            secret = root / "password"
            secret.write_text("synthetic-secret\n"); secret.chmod(0o600)
            old = os.environ["PATH"]
            os.environ["PATH"] = str(root) + os.pathsep + old
            try:
                result = sevenzip([], secret, root, timeout=15)
                self.assertEqual(len(archive_entries(result)), 50000)
                self.assertEqual(result.count("Path = "), 50000)
            finally:
                os.environ["PATH"] = old

    def test_stop_terminates_an_active_child_and_preserves_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            pid = work / "child.pid"

            def stop_soon():
                for _ in range(200):
                    if pid.exists():
                        (work / "STOP").touch()
                        return
                    time.sleep(.01)

            thread = threading.Thread(target=stop_soon)
            thread.start()
            command = [sys.executable, "-c", "import os,time,pathlib; pathlib.Path(__import__('sys').argv[1]).write_text(str(os.getpid())); time.sleep(10)", str(pid)]
            with self.assertRaises(Stopped):
                run_command(command, work, work / "child.log", timeout=5)
            thread.join()
            self.assertTrue((work / "STOP").exists())
            with self.assertRaises(ProcessLookupError):
                os.kill(int(pid.read_text()), 0)

    def test_range_resume_and_checksum(self):
        payload = b"synthetic model payload" * 100

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                start = int(self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
                self.send_response(206 if start else 200)
                self.send_header("Content-Length", str(len(payload) - start))
                if start:
                    self.send_header("Content-Range", f"bytes {start}-{len(payload)-1}/{len(payload)}")
                self.end_headers()
                self.wfile.write(payload[start:])

            def log_message(self, *_):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                target = work / "model.bin"
                target.with_name("model.bin.partial").write_bytes(payload[:100])
                download_file(f"http://127.0.0.1:{server.server_port}/model", target,
                              len(payload), hashlib.sha256(payload).hexdigest(), work, None)
                self.assertEqual(target.read_bytes(), payload)
                git_hash = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
                download_file(f"http://127.0.0.1:{server.server_port}/model", work / "git-model",
                              len(payload), git_hash, work, None, git_blob=True)
                target.write_bytes(b"modified")
                with self.assertRaises(ValueError):
                    download_file("unused", target, len(payload), hashlib.sha256(payload).hexdigest(), work, None)
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_password_is_sent_via_pty_without_echo(self):
        # Even with no data archive, test the exact terminal/password transport.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "7z"
            fake.write_text("#!/usr/bin/env python3\nimport getpass\np=getpass.getpass('Enter password: ')\nprint('accepted' if p=='synthetic-secret' else 'rejected')\n")
            fake.chmod(0o700)
            secret = root / "password"
            secret.write_text("synthetic-secret\n"); secret.chmod(0o600)
            old = os.environ["PATH"]
            os.environ["PATH"] = str(root) + os.pathsep + old
            try:
                result = sevenzip([], secret, root)
                self.assertIn("accepted", result)
                self.assertNotIn("synthetic-secret", result)
            finally:
                os.environ["PATH"] = old
