"""Small shared contracts: atomic artifacts, identity, STOP and subprocesses."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


class Stopped(RuntimeError):
    pass


class TransientFailure(RuntimeError):
    """Explicitly retryable transport failure, never auth/quota/data identity."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def stop_check(work: Path) -> None:
    if (work / "STOP").exists():
        raise Stopped(f"STOP present: {work / 'STOP'}")


def verified_marker(path: Path, fingerprint: str) -> bool:
    if not path.exists():
        return False
    state = json.loads(path.read_text())
    if state["fingerprint"] != fingerprint:
        raise RuntimeError(f"Identity changed: {path}; use a separate run, preserve this one")
    for name, sha in state.get("outputs", {}).items():
        target = path.parent / name
        if not target.is_file() or digest(target) != sha:
            raise RuntimeError(f"Incomplete or modified artifact: {target}")
    return True


def mark(path: Path, fingerprint: str, outputs: list[Path], **extra) -> None:
    atomic_json(path, {"fingerprint": fingerprint, "completed_at": now(),
                      "outputs": {os.path.relpath(p, path.parent): digest(p) for p in outputs}, **extra})


def run_command(command: list[str], work: Path, log: Path, timeout: float | None = None) -> int:
    """No shell; forward termination to the whole child group and retain logs."""
    stop_check(work)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open("a", buffering=1) as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        previous = {}

        def terminate(*_):
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

        def interrupted(*_):
            raise Stopped("signal")

        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, interrupted)
            while child.poll() is None:
                stop_check(work)
                if timeout is not None and time.monotonic() - started >= timeout:
                    raise TimeoutError(f"Stage time budget reached; log: {log}")
                time.sleep(0.5)
            return int(child.returncode)
        finally:
            terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def artifacts_fingerprint(root: Path) -> str:
    files = sorted(p for folder in ("kairos", "kairos_lab", "configs", "requirements")
                   for p in (root / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    files += [root / name for name in ("baseline.py", "evaluate.py") if (root / name).is_file()]
    return identity({str(p.relative_to(root)): digest(p) for p in files})
