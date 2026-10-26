r"""Download the official Kairos volumes, resume transfers, verify SHA-256.

No archive password or access code is stored by this script.
Run: .venv\Scripts\python.exe scripts/download_data.py --set all
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
from html.parser import HTMLParser
import http.cookiejar
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse, urlencode
from urllib.request import Request, urlopen, build_opener, HTTPCookieProcessor

ROOT = Path(__file__).resolve().parents[1]
PAGE = "https://kairos-labs.ai/dataset/"


def manifest():
    with urlopen(PAGE, timeout=60) as response:
        page = response.read().decode("utf-8")
    id_block = re.search(r"const DRIVE_IDS\s*=\s*\{(.*?)\};", page, re.S)
    volume_block = re.search(r"const VOLUMES\s*=\s*\[(.*?)\];", page, re.S)
    if not id_block or not volume_block:
        raise RuntimeError("Official download page layout changed")
    ids = dict(re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', id_block[1]))
    volumes = [dict(name=name, size=int(size), sha256=sha, drive_id=ids[name])
               for name, size, sha in re.findall(
                   r'\["([^"]+)",\s*(\d+),\s*"([0-9a-f]{64})"\]', volume_block[1])]
    if len(volumes) != 16:
        raise RuntimeError(f"Expected 16 official volumes, received {len(volumes)}")
    return volumes


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.params = {}
        self.in_form = False
        self.text = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id") == "download-form":
            self.action = attrs.get("action")
            self.in_form = True
        if tag == "input" and self.in_form and "name" in attrs:
            self.params[attrs["name"]] = attrs.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.in_form = False

    def handle_data(self, data):
        self.text.append(data)


def open_download(session, volume, offset):
    url = "https://drive.google.com/uc"
    params = {"export": "download", "id": volume["drive_id"]}
    headers = {"User-Agent": "Mozilla/5.0 Kairos-course-download/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    for _ in range(4):
        response = session.open(Request(url + "?" + urlencode(params), headers=headers), timeout=120)
        if "text/html" not in response.headers.get("Content-Type", ""):
            return response
        form = DownloadForm()
        form.feed(response.read(1024 * 1024).decode("utf-8", errors="replace"))
        response.close()
        if form.action is None:
            # Only a short server message, never a page full of private data.
            reason = " ".join(form.text)[-400:]
            raise RuntimeError(f"Google Drive did not provide a download: {reason}")
        url = form.action
        host = urlparse(url).hostname or ""
        if not (url.startswith("https://") and (host == "google.com" or host.endswith(".google.com"))):
            raise RuntimeError("Unexpected download confirmation host")
        params = form.params
    raise RuntimeError("Too many Google Drive confirmation pages")


def download(volume, directory):
    target = directory / volume["name"]
    partial = target.with_name(target.name + ".part")
    if target.exists():
        if target.stat().st_size == volume["size"] and checksum(target) == volume["sha256"]:
            print(f"VERIFIED {target.name} (already present)", flush=True)
            return
        raise RuntimeError(f"Existing {target.name} failed verification; kept for inspection")
    for attempt in range(1, 6):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > volume["size"]:
                raise RuntimeError("Partial file is larger than the official volume")
            if offset < volume["size"]:
                session = build_opener(HTTPCookieProcessor(http.cookiejar.CookieJar()))
                with open_download(session, volume, offset) as response:
                        if response.status == 206:
                            content_range = response.headers.get("Content-Range", "")
                            if not content_range.startswith(f"bytes {offset}-"):
                                raise RuntimeError("Server returned an incorrect byte range")
                        elif offset:
                            print(f"RESTART {target.name}: server does not support resume", flush=True)
                            offset = 0
                        last_report = time.monotonic()
                        with partial.open("ab" if offset else "wb") as destination:
                            for chunk in iter(lambda: response.read(4 * 1024 * 1024), b""):
                                destination.write(chunk)
                                offset += len(chunk)
                                if time.monotonic() - last_report > 30:
                                    print(f"DOWNLOAD {target.name} {offset / 1e9:.2f}/{volume['size'] / 1e9:.2f} GB", flush=True)
                                    last_report = time.monotonic()
            if partial.stat().st_size != volume["size"]:
                raise RuntimeError("Incomplete download")
            print(f"CHECKING {target.name}", flush=True)
            if checksum(partial) != volume["sha256"]:
                raise RuntimeError("SHA-256 mismatch; partial file kept for inspection")
            partial.replace(target)
            print(f"VERIFIED {target.name}", flush=True)
            return
        except (OSError, RuntimeError) as error:
            print(f"ATTEMPT {attempt} {target.name}: {error}", flush=True)
            if attempt == 5 or "SHA-256" in str(error):
                raise
            time.sleep(min(10 * attempt, 40))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=["core", "video", "all"], default="all")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    # Keep a second terminal from writing into the same partial volumes.
    lock_directory = ROOT / ".local"
    lock_directory.mkdir(exist_ok=True)
    download_lock = (lock_directory / "download.lock").open("a+b")
    download_lock.seek(0, 2)
    if download_lock.tell() == 0:
        download_lock.write(b"0")
        download_lock.flush()
    download_lock.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(download_lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(download_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit("Another dataset download is already running. Use check_status.cmd.") from None
    directory = ROOT / "archives"
    directory.mkdir(exist_ok=True)
    volumes = manifest()
    (directory / "manifest.json").write_text(json.dumps({"source": PAGE, "volumes": volumes}, indent=2), encoding="utf-8")
    selected = [v for v in volumes if args.set == "all" or v["name"].startswith(f"lab1_{args.set}.")]
    print(f"Official volumes: {len(selected)}, total {sum(v['size'] for v in selected) / 1e9:.2f} GB", flush=True)
    if args.manifest_only:
        return
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        pending = {pool.submit(download, volume, directory): volume for volume in selected}
        for future in concurrent.futures.as_completed(pending):
            try:
                future.result()
            except Exception as error:
                failures.append(pending[future]["name"])
                print(f"FAILED {pending[future]['name']}: {error}", flush=True)
    if failures:
        raise SystemExit("Failed volumes (rerun to resume): " + ", ".join(failures))
    print("All selected volumes passed SHA-256 verification.", flush=True)


if __name__ == "__main__":
    main()
