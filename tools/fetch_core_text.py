from __future__ import annotations

import argparse
from collections import OrderedDict
import getpass
from html.parser import HTMLParser
import io
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from urllib.parse import urlencode

import py7zr
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

VOLUMES = [
    ("1UEuzJOPCCnMlXW0atu9zbsAYdMNWhmXM", 10737418240),
    ("1YuGKAOXY1SUowWqBanxKnBHG7TB0UOo3", 10737418240),
    ("1sL-tE0Vn4aK76pDcbt0DhpTjCwzaYwyt", 10737418240),
    ("1BMSLb--9ATnX8r5q2u6HlqB3Dtdas7SA", 10737418240),
    ("1gG-7MP81j-LOUB7CdxU3Qci5R9cbMrW1", 7531318585),
]


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.params = {}

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "form" and a.get("id") == "download-form":
            self.action = a.get("action")
        if tag == "input" and a.get("type") == "hidden":
            self.params[a["name"]] = a.get("value", "")


class RemoteVolumes(io.RawIOBase):
    def __init__(self, max_bytes=8_000_000_000, chunk_size=1024 * 1024):
        super().__init__()
        self.position = 0
        self.size = sum(size for _, size in VOLUMES)
        self.downloaded = 0
        self.max_bytes = max_bytes
        self.chunk_size = chunk_size
        self.cache = OrderedDict()
        self.urls = {}
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        position = offset if whence == io.SEEK_SET else (
            self.position + offset if whence == io.SEEK_CUR else self.size + offset)
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def url(self, volume):
        if volume not in self.urls:
            identifier, _ = VOLUMES[volume]
            r = self.session.get("https://drive.google.com/uc", params={
                "export": "download", "id": identifier}, timeout=(15, 60), stream=True)
            try:
                r.raise_for_status()
                if "text/html" not in r.headers.get("Content-Type", ""):
                    self.urls[volume] = r.url
                else:
                    form = DownloadForm()
                    form.feed(r.raw.read(100_000).decode("utf-8"))
                    if not form.action or "google.com" not in form.action:
                        raise RuntimeError("Drive returned no official download form")
                    self.urls[volume] = form.action + "?" + urlencode(form.params)
            finally:
                r.close()
        return self.urls[volume]

    def chunk(self, volume, number):
        key = (volume, number)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        start = number * self.chunk_size
        end = min(start + self.chunk_size, VOLUMES[volume][1]) - 1
        expected = end - start + 1
        if self.downloaded + expected > self.max_bytes:
            raise RuntimeError("Download budget exceeded; archive may be solid. "
                               "Use local volumes or explicitly raise --max-download-gb.")
        with self.session.get(self.url(volume), headers={"Range": f"bytes={start}-{end}",
                                                        "Accept-Encoding": "identity"},
                              stream=True, timeout=(15, 120)) as r:
            r.raise_for_status()
            target = f"bytes {start}-{end}/{VOLUMES[volume][1]}"
            if r.status_code != 206 or r.headers.get("Content-Range") != target:
                raise RuntimeError("Server ignored or changed the byte range; full download refused")
            block = r.raw.read(expected + 1)
            if len(block) != expected:
                raise RuntimeError("Truncated or oversized byte range")
        self.downloaded += len(block)
        self.cache[key] = block
        if len(self.cache) > 32:
            self.cache.popitem(last=False)
        if self.downloaded // 100_000_000 != (self.downloaded - len(block)) // 100_000_000:
            print(f"Read {self.downloaded / 1e9:.2f} GB", flush=True)
        return block

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.position
        if size > 256 * 1024 * 1024:
            raise ValueError("Refusing a single read larger than 256 MiB")
        remaining = min(size, max(0, self.size - self.position))
        pieces = []
        while remaining:
            offset = self.position
            for volume, (_, volume_size) in enumerate(VOLUMES):
                if offset < volume_size:
                    break
                offset -= volume_size
            number, within = divmod(offset, self.chunk_size)
            block = self.chunk(volume, number)
            take = min(remaining, len(block) - within)
            pieces.append(block[within:within + take])
            remaining -= take
            self.position += take
        return b"".join(pieces)

    def close(self):
        self.session.close()
        super().close()


def wanted(name):
    p = PurePosixPath(name)
    if not p.parts:
        return False
    if p.is_absolute() or ".." in p.parts:
        raise ValueError("Unsafe archive path")
    parts = p.parts[1:] if p.parts and p.parts[0] in ["data", "lab1", "lab1_core"] else p.parts
    return (len(parts) == 1 and parts[0] in ["labels.csv", "split.json", "README.md"]
            or len(parts) == 2 and parts[0] in ["authors", "posts"] and parts[1].endswith(".json")
            or parts == ("media", "manifest.jsonl"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("data"))
    ap.add_argument("--max-download-gb", type=float, default=8)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--no-password", action="store_true", help="inspect an unencrypted header only")
    args = ap.parse_args()
    password = None if args.no_password else getpass.getpass("Official core archive password: ")
    args.output.mkdir(parents=True, exist_ok=True)
    with RemoteVolumes(max_bytes=int(args.max_download_gb * 1e9)) as stream:
        with py7zr.SevenZipFile(stream, password=password) as archive:
            names = archive.getnames()
            targets = [name for name in names if wanted(name)]
            target_set = set(targets)
            total = sum(f.uncompressed for f in archive.files if f.filename in target_set)
            print(f"Archive entries: {len(names)}; selected: {len(targets)}; "
                  f"selected unpacked: {total / 1e9:.3f} GB; solid: {archive._is_solid()}", flush=True)
            if not targets:
                raise RuntimeError("No expected release files found; inspect archive layout")
            if args.inspect:
                print("Selected examples:", targets[:8], flush=True)
                return
            if total + 1_000_000_000 > shutil.disk_usage(args.output).free:
                raise RuntimeError("Not enough free disk space for selected files")
            with tempfile.TemporaryDirectory(prefix=".extract-", dir=args.output) as temp:
                archive.extract(path=temp, targets=targets)
                root = Path(temp)
                for prefix in ["data", "lab1", "lab1_core"]:
                    if (root / prefix / "labels.csv").is_file():
                        root = root / prefix
                        break
                if not (root / "labels.csv").is_file() or not (root / "posts").is_dir():
                    raise RuntimeError("Archive does not contain the expected release layout")
                for source in root.iterdir():
                    target = args.output / source.name
                    if target.exists():
                        raise FileExistsError(f"Refusing to overwrite {target}")
                    shutil.move(str(source), str(target))
        print(f"Complete. Read {stream.downloaded / 1e9:.3f} GB; extracted file CRCs verified.")


if __name__ == "__main__":
    try:
        main()
    except py7zr.exceptions.PasswordRequired:
        raise SystemExit("The archive header is encrypted. Obtain the official password "
                         "at https://kairos-labs.ai/dataset/ and run without --no-password.")
