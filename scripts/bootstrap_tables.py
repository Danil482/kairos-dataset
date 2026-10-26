"""Read/extract only the non-media files via HTTP ranges while archives download.

The full archive downloads are still required for release SHA-256 verification.
Extracted streams are checked by py7zr where the archive provides CRCs.
Password: KAIROS_ARCHIVE_PASSWORD environment variable or an interactive prompt.
"""
from __future__ import annotations

import argparse
import bisect
import getpass
import http.cookiejar
import io
import json
import os
from pathlib import Path
from urllib.request import build_opener, HTTPCookieProcessor

from download_data import ROOT, open_download
import py7zr


class RemoteVolumes(io.BufferedIOBase):
    def __init__(self, volumes, max_remote_bytes):
        self.volumes = volumes
        self.ends = []
        for volume in volumes:
            self.ends.append((self.ends[-1] if self.ends else 0) + volume["size"])
        self.position = 0
        self.session = build_opener(HTTPCookieProcessor(http.cookiejar.CookieJar()))
        self.remote_bytes = 0
        self.max_remote_bytes = max_remote_bytes
        self.cache = ROOT / ".local" / "range-cache"
        self.cache.mkdir(parents=True, exist_ok=True)

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else (self.position if whence == 1 else self.ends[-1]) + offset
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def read(self, size=-1):
        if size < 0:
            size = self.ends[-1] - self.position
        size = min(size, self.ends[-1] - self.position)
        chunks = []
        while size > 0:
            number = bisect.bisect_right(self.ends, self.position)
            volume = self.volumes[number]
            offset = self.position - (self.ends[number - 1] if number else 0)
            block_size = 4 * 1024 * 1024
            block_start = offset // block_size * block_size
            block_end = min(block_start + block_size, volume["size"])
            take = min(size, block_end - offset)
            data = None
            for suffix in ("", ".part"):
                local = ROOT / "archives" / (volume["name"] + suffix)
                if local.exists() and local.stat().st_size >= offset + take:
                    with local.open("rb") as source:
                        source.seek(offset)
                        data = source.read(take)
                    break
            cache = self.cache / f"block-{number}-{block_start}.bin"
            if data is None and cache.exists():
                data = cache.read_bytes()[offset - block_start:offset - block_start + take]
            if data is None:
                fetch_size = block_end - block_start
                if self.remote_bytes + fetch_size > self.max_remote_bytes:
                    raise RuntimeError("Selective extraction exceeded its remote byte budget")
                with open_download(self.session, volume, block_start) as response:
                    content_range = response.headers.get("Content-Range", "")
                    if block_start and (response.status != 206 or not content_range.startswith(f"bytes {block_start}-")):
                        raise RuntimeError("Google Drive did not honor the byte range")
                    block = response.read(fetch_size)
                if len(block) != fetch_size:
                    raise RuntimeError("Incomplete ranged response")
                cache.write_bytes(block)
                self.remote_bytes += len(block)
                data = block[offset - block_start:offset - block_start + take]
                if self.remote_bytes // 50_000_000 != (self.remote_bytes - len(block)) // 50_000_000:
                    print(f"Table stream download: {self.remote_bytes / 1e6:.0f} MB", flush=True)
            chunks.append(data)
            self.position += len(data)
            size -= len(data)
        return b"".join(chunks)


def is_table(name):
    parts = Path(name).parts
    return any(part in ("authors", "posts") for part in parts) or Path(name).name in ("labels.csv", "split.json", "manifest.jsonl", "README.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--max-remote-gb", type=float, default=2.)
    args = parser.parse_args()
    password = os.environ.get("KAIROS_ARCHIVE_PASSWORD") or getpass.getpass("Archive password: ")
    volumes = json.loads((ROOT / "archives" / "manifest.json").read_text())["volumes"]
    volumes = [v for v in volumes if "core" in v["name"]]
    source = RemoteVolumes(volumes, int(args.max_remote_gb * 1e9))
    with py7zr.SevenZipFile(source, mode="r", password=password) as archive:
        targets = [name for name in archive.getnames() if is_table(name)]
        print(f"Archive entries: {len(archive.getnames())}; table entries: {len(targets)}", flush=True)
        packsizes = archive.header.main_streams.packinfo.packsizes
        print(f"Compression streams: {len(packsizes)}; first sizes: {packsizes[:8]}", flush=True)
        folders = archive.header.main_streams.unpackinfo.folders
        target_folders = {id(file.folder) for file in archive.files if is_table(file.filename) and not file.is_directory}
        print("Target stream indexes:", [i for i, folder in enumerate(folders) if id(folder) in target_folders], flush=True)
        if not args.inspect:
            for name in targets:
                path = Path(name)
                if path.is_absolute() or ".." in path.parts:
                    raise RuntimeError("Unsafe archive entry")
            archive.extract(path=ROOT / "data", targets=targets)
            print("Non-media files extracted.", flush=True)
    print(f"Fetched by range: {source.remote_bytes / 1e6:.2f} MB", flush=True)


if __name__ == "__main__":
    main()
