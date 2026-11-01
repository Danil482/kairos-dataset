"""Read validated frozen E5 vectors on CPU without loading Torch or the model.

This consumer never creates or repairs cache files. Producer metadata remains
unchanged: its CUDA/Python/library versions describe how the vectors were made,
not requirements for reading the resulting float32 array.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# This module imports only stdlib/NumPy/pandas at module scope. Its encoder entry
# point, which imports Torch lazily, is deliberately never called here.
from kairos.experiment_encode import history_fingerprint


PRODUCER_ENCODER_SHA256 = "ee894fa2dcd8b997b6496b5d0f7cbad7164962cf0eab4d847cc58b14d412b814"
FROZEN_IDENTITY = {
    "schema_version": 1,
    "model": "intfloat/multilingual-e5-small",
    "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
    "dimensions": 384,
    "prefix": "query: ",
    "token_budgets": {"own": 256, "shared": 160, "titles": 64},
    "encoder_sha256": PRODUCER_ENCODER_SHA256,
    "output_dtype": "float32",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_version(path: Path) -> tuple:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def load_frozen_embeddings(history: pd.DataFrame, cache_dir: Path) -> tuple[np.ndarray, dict]:
    """Return read-only vectors aligned exactly to ``history`` and producer metadata.

    Requires the completed, pinned 512-token E5 cache. Changed text/order,
    incompatible provenance, incomplete files and invalid arrays are errors;
    no model download, GPU operation or automatic overwrite is attempted.
    """
    expected_input = history_fingerprint(history)
    cache_dir = Path(cache_dir)
    metadata_path, array_path = cache_dir / "metadata.json", cache_dir / "history.npy"
    for path in (metadata_path, array_path):
        if not path.is_file():
            raise FileNotFoundError(f"Frozen embeddings cache is incomplete: missing {path}")
    metadata_version, array_version = _file_version(metadata_path), _file_version(array_path)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Cannot read frozen embeddings metadata: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError("Frozen embeddings metadata must be a JSON object")
    for key, expected in FROZEN_IDENTITY.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Frozen embeddings producer mismatch: {key}")
    config = metadata.get("config")
    if not isinstance(config, dict) or config.get("max_length") != 512:
        raise ValueError("Frozen embeddings require the pinned max_length=512 configuration")
    runtime = metadata.get("runtime")
    if not isinstance(runtime, dict) or not isinstance(runtime.get("versions"), dict):
        raise ValueError("Frozen embeddings are missing producer runtime provenance")
    # Deliberately do not query installed versions or require a CUDA device.
    if metadata.get("input") != expected_input:
        raise ValueError("Frozen embeddings history fingerprint mismatch: ordered keys or source text changed")
    if _sha256(array_path) != metadata.get("output_sha256"):
        raise ValueError("Frozen embeddings array checksum mismatch")
    try:
        vectors = np.load(array_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError("Frozen embeddings array is not a readable NumPy file") from exc
    if not isinstance(vectors, np.ndarray):
        if hasattr(vectors, "close"):
            vectors.close()
        raise ValueError("Frozen embeddings must contain one NumPy array")
    if vectors.shape != (len(history), 384):
        raise ValueError("Frozen embeddings array shape does not match history rows and 384 dimensions")
    if vectors.dtype != np.float32:
        raise ValueError("Frozen embeddings array dtype must be float32")
    for start in range(0, len(vectors), 8192):
        block = vectors[start:start + 8192]
        if not np.isfinite(block).all():
            raise ValueError("Frozen embeddings contain non-finite values")
        if not np.allclose(np.linalg.norm(block, axis=1), 1.0, atol=1e-4, rtol=1e-5):
            raise ValueError("Frozen embeddings are not unit-normalised")
    if (_file_version(metadata_path) != metadata_version or
            _file_version(array_path) != array_version):
        raise ValueError("Frozen embeddings cache changed while it was being validated")
    return vectors, metadata
