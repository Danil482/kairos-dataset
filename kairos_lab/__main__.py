"""Container CLI. Commands do not connect to any server."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

from .common import Stopped, TransientFailure, artifacts_fingerprint, atomic_json, identity, now, stop_check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "extract", "fetch-models", "prepare", "text", "images", "train", "score", "verify", "manifest"])
    parser.add_argument("--config", type=Path, default=Path("/app/configs/night.json"))
    parser.add_argument("--work", type=Path, default=Path("/work"))
    parser.add_argument("--data", type=Path, default=Path("/inputs"))
    parser.add_argument("--downloads", type=Path, default=Path("/downloads"))
    parser.add_argument("--models", type=Path, default=Path("/models"))
    parser.add_argument("--secret", type=Path, default=Path("/run/secrets/archive-password"))
    parser.add_argument("--manifest", type=Path, default=Path("/app/docs/core-downloads.tsv"))
    parser.add_argument("--night-until", default=os.environ.get("KAIROS_NIGHT_UNTIL"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    if os.environ.get("KAIROS_DEADLINE"):
        config["deadline"] = os.environ["KAIROS_DEADLINE"]
    stop_check(args.work)
    root = Path(__file__).resolve().parent.parent
    env_path = args.work / "environment/manifest.json"
    fingerprint = identity({"code": artifacts_fingerprint(root), "config": config})
    versions = {}
    for name in ("numpy", "pandas", "scikit-learn", "catboost", "gdown", "torch", "transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    if env_path.exists():
        previous = json.loads(env_path.read_text())
        if previous["fingerprint"] != fingerprint:
            raise ValueError("Code/config changed within run; resume is refused")
    else:
        atomic_json(env_path, {"fingerprint": fingerprint, "created": now(), "config": config, "versions": versions})
    stage = args.command
    atomic_json(args.work / "environment" / f"runtime-{stage}.json", {"at": now(), "versions": versions,
                                                                     "python": sys.version, "uid": os.getuid()})
    atomic_json(args.work / "status.json", {"state": "running", "stage": stage, "at": now()})
    try:
        if stage in ("download", "extract", "fetch-models"):
            from .acquire import download_core, extract, fetch_models
            if stage == "download":
                download_core(args.manifest, args.downloads, args.work, args.night_until)
            elif stage == "extract":
                release = extract(args.downloads, args.data, args.secret, args.work)
                atomic_json(args.work / "release-root.json", {"relative": str(release.relative_to(args.data))})
            else:
                fetch_models(args.models, args.work, config, args.night_until)
        else:
            release_info = args.work / "release-root.json"
            data = args.data / json.loads(release_info.read_text())["relative"] if release_info.exists() else args.data
            if stage == "prepare":
                from .dataset import prepare
                prepare(data, args.work, config)
            elif stage in ("text", "images"):
                from .encoders import MediaSchemaError, encode_images, encode_text
                try:
                    if stage == "text":
                        encode_text(args.models, args.work, config, args.device)
                    elif config["enable_images"]:
                        encode_images(data, args.models, args.work, config, args.device)
                except MediaSchemaError as exc:
                    atomic_json(args.work / "optional-images-skipped.json", {"reason": str(exc), "at": now()})
                    atomic_json(args.work / "status.json", {"state": "needs_attention", "stage": stage,
                                                            "reason": str(exc), "at": now()})
                    return 20
            elif stage == "train":
                from .train import train_models
                train_models(args.work, config)
            elif stage == "score":
                from .train import score_final
                score_final(data, args.work, config)
            elif stage == "verify":
                from .verify import verify
                verify(args.work, config)
        atomic_json(args.work / "status.json", {"state": "completed", "stage": stage, "at": now()})
        return 0
    except Stopped as exc:
        atomic_json(args.work / "status.json", {"state": "stopped", "stage": stage, "reason": str(exc), "at": now()})
        return 130
    except Exception as exc:
        retryable = isinstance(exc, TransientFailure)
        if stage in ("download", "fetch-models"):
            import requests
            retryable |= isinstance(exc, (requests.Timeout, requests.ConnectionError)) and not isinstance(exc, requests.exceptions.SSLError)
        atomic_json(args.work / "status.json", {"state": "needs_attention", "stage": stage,
                                                "error_type": type(exc).__name__, "reason": str(exc),
                                                "retryable": retryable, "at": now()})
        if retryable:
            return 75
        raise


if __name__ == "__main__":
    sys.exit(main())
