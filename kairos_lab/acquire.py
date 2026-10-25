"""Bounded public downloads and password-safe, validated archive extraction."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import requests

from .common import TransientFailure, atomic_json, digest, identity, mark, run_command, stop_check, verified_marker


def night_remaining(until: str | None) -> float:
    if not until:
        return 0.0
    dt = datetime.fromisoformat(until)
    if dt.tzinfo is None:
        raise ValueError("night-until must include a timezone")
    return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())


def model_metadata(url: str, work: Path) -> dict:
    for attempt in range(4):
        stop_check(work)
        try:
            response = requests.get(url, params={"blobs": "true"}, timeout=(30, 60))
            response.raise_for_status()
            return response.json()
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if attempt == 3 or isinstance(exc, requests.exceptions.SSLError) or (status and status not in (408, 500, 502, 503, 504)):
                raise
            for _ in range(2**attempt):
                stop_check(work)
                time.sleep(1)


def download_core(manifest: Path, downloads: Path, work: Path, night_until: str | None) -> None:
    downloads.mkdir(parents=True, exist_ok=True)
    with manifest.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    for row in rows:
        stop_check(work)
        target = downloads / row["filename"]
        expected_size = int(row["bytes"])
        if target.is_file() and target.stat().st_size == expected_size:
            if digest(target) != row["sha256"]:
                raise ValueError(f"Checksum mismatch; preserve and inspect: {target}")
            continue
        command = [sys.executable, "-m", "gdown", row["url"], "--no-cookies", "--continue",
                   "--retries", "3", "--timeout", "60", "--output", str(target)]
        remaining = night_remaining(night_until)
        if remaining:
            try:
                code = run_command(command, work, work / "logs/download.log", timeout=remaining)
            except TimeoutError:
                code = run_command(command + ["--speed", "5000KB"], work, work / "logs/download.log")
        else:
            code = run_command(command + ["--speed", "5000KB"], work, work / "logs/download.log")
        if code or not target.is_file() or target.stat().st_size != expected_size:
            log = (work / "logs/download.log").read_text(errors="replace")[-6000:]
            fatal = re.search(r"(?i)quota|too many users|permission|access denied|forbidden|certificate|403|401|429", log)
            transient = re.search(r"(?i)ReadTimeout|ConnectionError|Connection reset|RemoteDisconnected|timed out|IncompleteRead|502|503|504", log)
            if transient and not fatal:
                raise TransientFailure(f"Transient core download failure: {row['filename']}; partial preserved")
            raise RuntimeError(f"Download incomplete: {row['filename']}; see logs/download.log")
        if digest(target) != row["sha256"]:
            raise ValueError(f"Checksum mismatch: {target}")
        atomic_json(work / "download-status.json", {"last_verified": row["filename"], "sha256": row["sha256"]})


def sevenzip(args: list[str], secret: Path, work: Path, timeout: float = 7200) -> str:
    """Password is sent through a no-echo PTY, never argv/environment/log files."""
    import pexpect
    if secret.stat().st_mode & 0o077:
        raise PermissionError("Archive password file must have mode 0600")
    password = secret.read_text().rstrip("\r\n")
    if not password or "\n" in password or "\r" in password:
        raise ValueError("Password file must contain one nonempty line")
    child = pexpect.spawn("7z", args, encoding="utf-8", codec_errors="replace", echo=False, timeout=1,
                          maxread=65536)
    child.delaybeforesend = 0.1
    output, prompts, tail = [], 0, ""
    started = time.monotonic()
    try:
        while True:
            stop_check(work)
            if time.monotonic() - started > timeout:
                raise TimeoutError("7z time budget exceeded")
            # Streaming reads avoid expect() repeatedly rescanning and copying the
            # entire multi-megabyte -slt listing while waiting for EOF.
            try:
                chunk = child.read_nonblocking(size=65536, timeout=1)
            except pexpect.TIMEOUT:
                continue
            except pexpect.EOF:
                break
            output.append(chunk)
            tail = (tail + chunk)[-2048:]
            if re.search(r"(?i)enter password[^:]*:", tail):
                prompts += 1
                if prompts > 1:
                    raise ValueError("7z rejected password or requested it repeatedly")
                child.setecho(False)
                child.sendline(password)
                tail = ""
        child.close()
        if child.exitstatus != 0:
            # Raw 7z output may contain filenames; retain only a bounded, redacted diagnostic.
            diagnostic = "".join(output).replace(password, "[REDACTED]")[-2000:]
            raise RuntimeError(f"7z failed ({child.exitstatus}): {diagnostic}")
        return "".join(output).replace(password, "[REDACTED]")
    finally:
        if child.isalive():
            child.terminate(force=True)


def archive_entries(listing: str) -> list[dict]:
    """Parse 7z -slt entries after its archive metadata separator."""
    if "----------" not in listing:
        raise ValueError("Could not parse 7z archive listing")
    result = []
    for block in listing.split("----------", 1)[1].replace("\r", "").split("\n\n"):
        fields = dict(line.split(" = ", 1) for line in block.splitlines() if " = " in line)
        if "Path" not in fields:
            continue
        path = PurePosixPath(fields["Path"].replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or ":" in str(path):
            raise ValueError("Unsafe archive path")
        if fields.get("Symbolic Link") or fields.get("Hard Link") or "L" in fields.get("Attributes", ""):
            raise ValueError("Links in the release archive are not supported")
        result.append(fields)
    if not result:
        raise ValueError("Archive contains no recognized entries")
    return result


def extract(downloads: Path, inputs: Path, secret: Path, work: Path) -> Path:
    inputs.mkdir(parents=True, exist_ok=True)
    volumes = sorted(downloads.glob("lab1_core.7z.*"))
    fingerprint = identity({p.name: digest(p) for p in volumes})
    marker = inputs / ".extracted.json"
    if verified_marker(marker, fingerprint):
        root = inputs / json.loads(marker.read_text())["release_root"]
        if not (root / "posts").is_dir() or not (root / "authors").is_dir():
            raise ValueError("Extracted dataset is incomplete")
        return root
    if len(volumes) != 5:
        raise ValueError("All five core volumes are required")
    listing = sevenzip(["l", "-slt", str(volumes[0])], secret, work)
    entries = archive_entries(listing)
    size = sum(int(e.get("Size", 0)) for e in entries)
    if shutil.disk_usage(inputs).free < size * 1.20 + 10 * 1024**3:
        raise RuntimeError(f"Need {size * 1.20 / 1024**3 + 10:.1f} GiB free for extraction plus reserve")
    atomic_json(work / "archive-audit.json", {"entries": len(entries), "unpacked_bytes": size})
    sevenzip(["x", "-y", "-bb0", "-bd", str(volumes[0]), "-o" + str(inputs)], secret, work)
    candidates = [p.parent for p in inputs.rglob("labels.csv")
                  if (p.parent / "posts").is_dir() and (p.parent / "authors").is_dir()]
    if len(candidates) != 1:
        raise ValueError("Expected exactly one release root with labels, posts and authors")
    root = candidates[0]
    for path in inputs.rglob("*"):
        if path.is_symlink() or not path.resolve().is_relative_to(inputs.resolve()):
            raise ValueError("Unsafe extracted path")
    mark(marker, fingerprint, [root / "labels.csv", root / "split.json"], release_root=str(root.relative_to(inputs)))
    return root


def download_file(url: str, path: Path, expected: int, sha: str, work: Path, night_until: str | None,
                  git_blob: bool = False) -> None:
    """Sequential HTTP with checked Range resume, SHA-256 and a day/night limiter."""
    def checksum(target):
        if not git_blob:
            return digest(target)
        h = hashlib.sha1(f"blob {target.stat().st_size}\0".encode())
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    if path.exists():
        if path.stat().st_size == expected and checksum(path) == sha:
            return
        raise ValueError(f"Modified model file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".partial")
    for attempt in range(4):
        stop_check(work)
        offset = part.stat().st_size if part.exists() else 0
        if offset == expected:
            break
        try:
            with requests.get(url, headers={"Range": f"bytes={offset}-"} if offset else {},
                              stream=True, timeout=(30, 60)) as response:
                response.raise_for_status()
                if offset and (response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                    raise ValueError("Server refused exact resume; partial preserved")
                tick, transferred = time.monotonic(), 0
                with part.open("ab" if offset else "wb") as stream:
                    for block in response.iter_content(256 * 1024):
                        stop_check(work)
                        stream.write(block)
                        if not night_remaining(night_until):
                            transferred += len(block)
                            delay = transferred / (5000 * 1024) - (time.monotonic() - tick)
                            if delay > 0:
                                time.sleep(delay)
                        else:
                            tick, transferred = time.monotonic(), 0
            if part.stat().st_size == expected:
                break
            raise requests.ConnectionError("Incomplete model download")
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if attempt == 3 or status in (401, 403, 404, 429) or isinstance(exc, requests.exceptions.SSLError):
                raise
            for _ in range(2**attempt):
                stop_check(work)
                time.sleep(1)
    if part.stat().st_size != expected or checksum(part) != sha:
        raise ValueError(f"Model checksum/size mismatch: {part}")
    part.replace(path)


def fetch_models(models: Path, work: Path, config: dict, night_until: str | None) -> None:
    for role in ("text_encoder", "image_encoder"):
        if role == "image_encoder" and not config["enable_images"]:
            continue
        settings = config[role]
        repo, revision = settings["repo"], settings["revision"]
        folder = models / role / revision
        marker = folder / "verified.json"
        fingerprint = identity(settings)
        if verified_marker(marker, fingerprint):
            continue
        metadata = model_metadata(f"https://huggingface.co/api/models/{repo}/revision/{revision}", work)
        if metadata["sha"] != revision:
            raise ValueError("Model revision identity mismatch")
        siblings = {x["rfilename"]: x for x in metadata["siblings"]}
        outputs = []
        for filename in settings["files"]:
            stop_check(work)
            info = siblings[filename]
            path = folder / filename
            url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
            if "lfs" in info:
                download_file(url, path, info["lfs"]["size"], info["lfs"]["sha256"], work, night_until)
            else:
                download_file(url, path, info["size"], info["blobId"], work, night_until, git_blob=True)
            outputs.append(path)
        mark(marker, fingerprint, outputs, repo=repo, revision=revision)
