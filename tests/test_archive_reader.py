from __future__ import annotations

import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import py7zr

from tools import fetch_core_text as reader


class ArchiveReaderTests(unittest.TestCase):
    def test_encrypted_multipart_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            (source / "posts").mkdir(parents=True)
            (source / "media").mkdir()
            (source / "labels.csv").write_text("author,post,split,label\nA,P,train,1\n")
            (source / "posts" / "AUTHOR_1.json").write_text('{"author":"A","posts":[]}')
            (source / "media" / "unused.jpg").write_bytes(b"not needed")
            archive_path = root / "fixture.7z"
            with py7zr.SevenZipFile(archive_path, "w", password="fixture-only",
                                    header_encryption=True) as archive:
                for path in sorted(source.rglob("*")):
                    if path.is_file():
                        archive.write(path, arcname=str(path.relative_to(source)))
            content = archive_path.read_bytes()
            middle = len(content) // 2
            chunks = [content[:middle], content[middle:]]

            class MemoryVolumes(reader.RemoteVolumes):
                def chunk(self, volume, number):
                    start = number * self.chunk_size
                    return chunks[volume][start:start + self.chunk_size]

            with patch.object(reader, "VOLUMES", [("first", middle), ("second", len(content) - middle)]):
                with MemoryVolumes(chunk_size=31) as stream:
                    self.assertEqual(stream.read(32), content[:32])
                    stream.seek(-17, io.SEEK_END)
                    self.assertEqual(stream.read(32), content[-17:])
                    self.assertEqual(stream.read(1), b"")
                    stream.seek(0)
                    with py7zr.SevenZipFile(stream, password="fixture-only") as archive:
                        targets = [p for p in archive.getnames() if reader.wanted(p)]
                        archive.extract(path=root / "out", targets=targets)
            self.assertEqual((root / "out" / "labels.csv").read_bytes(),
                             (source / "labels.csv").read_bytes())
            self.assertTrue((root / "out" / "posts" / "AUTHOR_1.json").is_file())
            self.assertFalse((root / "out" / "media" / "unused.jpg").exists())

    def test_unsafe_paths_refused(self):
        for name in ["../labels.csv", "/labels.csv", "posts/../../x.json"]:
            with self.assertRaises(ValueError):
                reader.wanted(name)
        self.assertFalse(reader.wanted(""))


if __name__ == "__main__":
    unittest.main()
