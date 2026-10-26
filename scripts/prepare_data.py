"""Finish downloading, verify every volume, and extract core/video automatically.

The password is held in process memory and is never written into the repository.
Use --wait-for-downloads if download_data.py is already running.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from download_data import ROOT, checksum, manifest

STATUS = ROOT / ".local" / "setup_status.json"


def downloader_running():
    """Probe the downloader's OS lock without relying on stale PID files."""
    path = ROOT / ".local" / "download.lock"
    path.parent.mkdir(exist_ok=True)
    with path.open("a+b") as lock:
        lock.seek(0, 2)
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(lock, fcntl.LOCK_UN)
        except OSError:
            return True
    return False


def save_status(**state):
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    STATUS.parent.mkdir(exist_ok=True)
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(STATUS)


def extract(kind, volumes, password, sevenzip):
    for volume in volumes:
        save_status(stage="verifying", archive=kind, volume=volume["name"])
        print(f"Checking SHA-256: {volume['name']}", flush=True)
        path = ROOT / "archives" / volume["name"]
        if path.stat().st_size != volume["size"] or checksum(path) != volume["sha256"]:
            raise RuntimeError(f"Volume failed SHA-256 verification: {volume['name']}")
    print(f"{kind}: all volume hashes match; extracting...", flush=True)
    save_status(stage="extracting", archive=kind)
    # -aos preserves previously bootstrapped tables; 7-Zip checks extracted data.
    command = [sevenzip, "x", str(ROOT / "archives" / f"lab1_{kind}.7z.001"),
               f"-o{ROOT / 'data'}", "-aos", "-bsp0", "-bb0", f"-p{password}"]
    # Do not use check=True: CalledProcessError would print the password argument.
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"7-Zip failed for {kind}, exit code {result.returncode}")
    receipt = dict(archive=kind, volumes=[dict(name=v["name"], sha256=v["sha256"]) for v in volumes],
                   extracted_at_utc=datetime.now(timezone.utc).isoformat())
    (ROOT / ".local" / f"verified_{kind}.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-downloads", action="store_true")
    parser.add_argument("--sevenzip", default=shutil.which("7z") or r"C:\Program Files\7-Zip\7z.exe")
    args = parser.parse_args()
    # Keep repeated double-clicks from launching two archive extractors.
    STATUS.parent.mkdir(exist_ok=True)
    setup_lock = (STATUS.parent / "prepare.lock").open("a+b")
    setup_lock.seek(0, 2)
    if setup_lock.tell() == 0:
        setup_lock.write(b"0")
        setup_lock.flush()
    setup_lock.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(setup_lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(setup_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit("Dataset preparation is already running. Use check_status.cmd.") from None
    password = os.environ.pop("KAIROS_ARCHIVE_PASSWORD", None) or getpass.getpass("Archive password: ")
    if not Path(args.sevenzip).is_file():
        raise SystemExit("7-Zip not found; supply --sevenzip with its executable path")
    manifest_path = ROOT / "archives" / "manifest.json"
    if manifest_path.exists():
        volumes = json.loads(manifest_path.read_text())["volumes"]
    else:
        volumes = manifest()
    child = None
    if not args.wait_for_downloads:
        if downloader_running():
            print("Download is already running; waiting for it and resuming extraction.", flush=True)
        else:
            child = subprocess.Popen([sys.executable, "-u", "scripts/download_data.py", "--set", "all"], cwd=ROOT)
    try:
        for kind in ("core", "video"):
            selected = [v for v in volumes if v["name"].startswith(f"lab1_{kind}.")]
            if (ROOT / ".local" / f"verified_{kind}.json").exists():
                continue
            print(f"Waiting for all {kind} volumes...", flush=True)
            last_progress = time.monotonic()
            previous_bytes = -1
            while True:
                missing = [v["name"] for v in selected if not (ROOT / "archives" / v["name"]).is_file()]
                if not missing:
                    break
                save_status(stage="downloading", archive=kind, missing=missing)
                current_bytes = sum(p.stat().st_size for p in (ROOT / "archives").glob("lab1_*.7z.*"))
                if current_bytes != previous_bytes:
                    last_progress = time.monotonic()
                    previous_bytes = current_bytes
                elif time.monotonic() - last_progress > 1800:
                    raise RuntimeError("Downloads have made no progress for 30 minutes; check the connection and resume")
                if child is not None and child.poll() is not None:
                    raise RuntimeError("Download process ended before all volumes were ready; rerun to resume")
                time.sleep(30)
            extract(kind, selected, password, args.sevenzip)
        if child is not None:
            child.wait()
        save_status(stage="complete", message="All 16 volumes verified and both archives extracted")
        print("Complete: all archives verified and extracted.", flush=True)
    except Exception as error:
        save_status(stage="error", message=str(error))
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
