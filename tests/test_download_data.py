import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import download_data


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}


class DownloadTests(unittest.TestCase):
    def run_transfer(self, prefix, response):
        body = b"0123456789" * 100
        volume = dict(name="sample.7z.001", size=len(body), sha256=hashlib.sha256(body).hexdigest())
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            partial = directory / (volume["name"] + ".part")
            partial.write_bytes(prefix)
            with patch.object(download_data, "open_download", return_value=response):
                download_data.download(volume, directory)
            self.assertEqual((directory / volume["name"]).read_bytes(), body)
            self.assertFalse(partial.exists())

    def test_range_resume_produces_exact_file(self):
        body = b"0123456789" * 100
        self.run_transfer(body[:100], Response(body[100:], 206, {"Content-Range": "bytes 100-999/1000"}))

    def test_server_without_ranges_restarts_instead_of_appending(self):
        body = b"0123456789" * 100
        self.run_transfer(body[:100], Response(body, 200))

    def test_corruption_is_never_promoted_to_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            volume = dict(name="broken.7z.001", size=3, sha256=hashlib.sha256(b"abc").hexdigest())
            with patch.object(download_data, "open_download", return_value=Response(b"bad")):
                with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                    download_data.download(volume, directory)
            self.assertFalse((directory / volume["name"]).exists())

    def test_google_confirmation_form(self):
        form = download_data.DownloadForm()
        form.feed('<form id="download-form" action="https://drive.usercontent.google.com/download">'
                  '<input name="id" value="sample"><input name="confirm" value="t"></form>')
        self.assertEqual(form.params, {"id": "sample", "confirm": "t"})


if __name__ == "__main__":
    unittest.main()
