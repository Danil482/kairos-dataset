"""Encode the prepared posts once with a frozen multilingual text model.

    python encode_text.py --device cuda

The CSV row order is preserved. metadata.json binds the arrays to the exact
input files, model revision and settings; test labels are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
DIMENSIONS = 384
PREFIX = "query: "
ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cached(output: Path, expected: dict) -> bool:
    """Only reuse a complete, matching and uncorrupted pair of arrays."""
    try:
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        if any(metadata.get(key) != value for key, value in expected.items()):
            return False
        for split, count in expected["rows"].items():
            path = output / f"{split}.npy"
            if sha256(path) != metadata["output_sha256"][split]:
                return False
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.shape != (count, DIMENSIONS) or array.dtype != np.float32:
                return False
            if not np.isfinite(array).all():
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/prepared")
    parser.add_argument("--output", type=Path, default=ROOT / "data/embeddings")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512,
                        help="token limit, including query prefix; longer posts are truncated")
    args = parser.parse_args()
    if args.batch_size < 1 or not 1 <= args.max_length <= 512:
        parser.error("batch-size must be positive and max-length must be in 1..512")

    # Delay GPU imports so --help works with only the baseline dependencies.
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("Install requirements-gpu.txt in the PyTorch environment first") from exc

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; check the GPU environment")
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but unavailable")

    paths = {split: args.prepared / f"{split}.csv.gz" for split in ("train", "test")}
    hashes = {split: sha256(path) for split, path in paths.items()}
    frames = {}
    for split, path in paths.items():
        frame = pd.read_csv(path, usecols=["author", "post", "text"],
                            dtype=str, keep_default_na=False)
        if frame.empty or frame.duplicated(["author", "post"]).any():
            raise SystemExit(f"{split}: expected nonempty input with unique post keys")
        if frame[["author", "post"]].eq("").any().any():
            raise SystemExit(f"{split}: post keys must not be empty")
        frames[split] = frame
    if set(frames["train"]["author"]) & set(frames["test"]["author"]):
        raise SystemExit("an author occurs in both train and test")

    versions = {name: importlib.metadata.version(name)
                for name in ("torch", "sentence-transformers", "transformers", "numpy")}
    expected = {
        "schema_version": 1,
        "model": MODEL,
        "revision": REVISION,
        "input_sha256": hashes,
        "rows": {split: len(frame) for split, frame in frames.items()},
        "dimensions": DIMENSIONS,
        "settings": {"max_length": args.max_length, "batch_size": args.batch_size,
                     "device": device, "prefix": PREFIX, "normalize_embeddings": True,
                     "dtype": "float32", "versions": versions},
    }
    if cached(args.output, expected):
        print("Complete matching embeddings already exist; nothing to encode.", flush=True)
        return

    args.output.mkdir(parents=True, exist_ok=True)
    # Metadata is the completion marker. Remove a stale marker before replacing arrays.
    (args.output / "metadata.json").unlink(missing_ok=True)
    print(f"Loading {MODEL} at {REVISION} on {device}; max tokens {args.max_length}",
          flush=True)
    model = SentenceTransformer(MODEL, revision=REVISION, device=device,
                                trust_remote_code=False)
    model.max_seq_length = args.max_length
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    output_hashes = {}
    timings = {}
    for split, frame in frames.items():
        started = time.monotonic()
        print(f"Encoding {split}: {len(frame):,} posts", flush=True)
        # E5's model card prescribes query: for classification, in every language.
        # Explicit prompt overrides any prompt saved with the model.
        with torch.inference_mode():
            embeddings = model.encode(
                frame["text"].replace("", "__empty__").tolist(), prompt=PREFIX,
                batch_size=args.batch_size, show_progress_bar=True,
                convert_to_numpy=True, normalize_embeddings=True,
            )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.shape != (len(frame), DIMENSIONS) or not np.isfinite(embeddings).all():
            raise SystemExit(f"{split}: invalid embedding dimensions or non-finite values")
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise SystemExit(f"{split}: embeddings are not unit-normalized")
        path = args.output / f"{split}.npy"
        temporary = path.with_suffix(".npy.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, embeddings, allow_pickle=False)
        temporary.replace(path)
        output_hashes[split] = sha256(path)
        timings[split] = round(time.monotonic() - started, 3)
        print(f"Saved {split}: {embeddings.shape}, {timings[split]:.1f}s", flush=True)
        del embeddings

    if any(sha256(paths[split]) != hashes[split] for split in paths):
        raise SystemExit("Prepared files changed during encoding; rerun with stable inputs")
    metadata = {**expected, "output_sha256": output_hashes, "seconds": timings}
    temporary = args.output / "metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output / "metadata.json")
    print("Both splits complete; metadata.json records input and output checksums.", flush=True)


if __name__ == "__main__":
    main()
