"""Frozen, ordered history embeddings with bounded multi-source token input.

``encode_history`` includes unlabelled early posts as context. It reads only the
explicit key/text columns below, never labels or engagement counters. GPU/model
dependencies are imported only when the function is called.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
import pandas as pd

MODEL = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
DIMENSIONS = 384
PREFIX = "query: "
KEY_COLUMNS = ("author", "post")
TEXT_COLUMNS = ("text", "content_text", "shared_text", "article_title",
                "document_title", "document_text")
SOURCE_NAMES = ("own", "shared", "titles", "document")
BASE_BUDGETS = (256, 160, 64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_hash(records) -> str:
    digest = hashlib.sha256()
    for row in records:
        encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def history_fingerprint(history: pd.DataFrame) -> dict:
    """Bind vectors to ordered post keys and all supplied source text fields."""
    required = list(KEY_COLUMNS + TEXT_COLUMNS)
    missing = set(required) - set(history.columns)
    if missing:
        raise ValueError(f"history is missing columns: {sorted(missing)}")
    if history.empty:
        raise ValueError("history must contain at least one post")
    keys = history.loc[:, list(KEY_COLUMNS)]
    if keys.isna().any().any() or keys.astype(str).eq("").any().any():
        raise ValueError("history post keys must not be missing or empty")
    if keys.duplicated().any():
        raise ValueError("history has duplicate (author, post) keys")
    text = history.loc[:, list(TEXT_COLUMNS)].fillna("").astype(str)
    return {
        "rows": len(history),
        "key_columns": list(KEY_COLUMNS),
        "text_columns": list(TEXT_COLUMNS),
        "ordered_keys_sha256": _record_hash(keys.astype(str).itertuples(index=False, name=None)),
        "ordered_text_sha256": _record_hash(text.itertuples(index=False, name=None)),
    }


def _configuration(config: dict) -> dict:
    defaults = {"device": "cpu", "batch_size": 64, "max_length": 512,
                "bf16": True, "show_progress": True}
    unknown = set(config) - set(defaults)
    if unknown:
        raise ValueError(f"unknown encoder settings: {sorted(unknown)}")
    settings = {**defaults, **config}
    if settings["device"] not in {"cuda", "cpu", "mps"}:
        raise ValueError("device must be explicitly cuda, cpu, or mps")
    for key, maximum in (("batch_size", None), ("max_length", 512)):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
        if maximum is not None and value > maximum:
            raise ValueError(f"{key} must not exceed {maximum}")
    for key in ("bf16", "show_progress"):
        if not isinstance(settings[key], bool):
            raise ValueError(f"{key} must be boolean")
    return settings


def _sources(row: tuple) -> list[str]:
    own, _combined, shared, article_title, document_title, document = row
    titles = "\n".join(value for value in (article_title, document_title) if value)
    result = [own, shared, titles, document]
    if not any(result):
        result[0] = "__empty__"
    return result


def _prepare_input_batch(rows: list[tuple], tokenizer, max_length: int):
    """Pack exact token IDs: own 256, shared 160, titles 64, document remainder.

    Unused allocations flow to the next source. Prefix and special tokens count
    against the total limit. No decode/re-tokenize step can change that limit.
    """
    prefix_ids = tokenizer.encode(PREFIX, add_special_tokens=False)
    special = tokenizer.num_special_tokens_to_add(pair=False)
    capacity = max_length - len(prefix_ids) - special
    if capacity < 1:
        raise ValueError("max_length leaves no room after the E5 prefix and special tokens")
    flat = [source for row in rows for source in _sources(row)]
    encoded = tokenizer(flat, add_special_tokens=False, truncation=False,
                        return_attention_mask=False, return_token_type_ids=False,
                        verbose=False)["input_ids"]
    features, stats = [], {"rows": len(rows), "truncated_rows": 0,
                           "input_tokens": 0, "max_input_tokens": 0,
                           "source_tokens": dict.fromkeys(SOURCE_NAMES, 0),
                           "kept_tokens": dict.fromkeys(SOURCE_NAMES, 0),
                           "truncated_by_source": dict.fromkeys(SOURCE_NAMES, 0)}
    for offset in range(0, len(encoded), len(SOURCE_NAMES)):
        remaining, carry, clipped = capacity, 0, False
        input_ids = list(prefix_ids)
        for index, (name, ids) in enumerate(zip(SOURCE_NAMES, encoded[offset:offset + 4])):
            allocation = min(remaining, BASE_BUDGETS[index] + carry) if index < 3 else remaining
            kept = min(len(ids), allocation)
            input_ids.extend(ids[:kept])
            remaining -= kept
            carry = allocation - kept
            stats["source_tokens"][name] += len(ids)
            stats["kept_tokens"][name] += kept
            if kept < len(ids):
                clipped = True
                stats["truncated_by_source"][name] += 1
        feature = tokenizer.prepare_for_model(input_ids, add_special_tokens=True,
                                               truncation=False, return_attention_mask=True)
        length = len(feature["input_ids"])
        if length > max_length:
            raise ValueError("tokenizer exceeded the requested sequence length")
        features.append(feature)
        stats["truncated_rows"] += int(clipped)
        stats["input_tokens"] += length
        stats["max_input_tokens"] = max(stats["max_input_tokens"], length)
    return features, stats


def _normalise(vectors: np.ndarray, rows: int) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    if values.shape != (rows, DIMENSIONS) or not np.isfinite(values).all():
        raise ValueError("encoder returned invalid dimensions or non-finite values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("encoder returned a zero vector")
    # BF16 pooling/normalisation can round norms; always normalise in float32.
    values = values / norms
    if not np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError("encoder vectors are not unit-normalised")
    return values


def _load_cache(output: Path, expected: dict):
    try:
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            return None
        if any(metadata.get(key) != value for key, value in expected.items()):
            return None
        path = output / "history.npy"
        if _sha256(path) != metadata["output_sha256"]:
            return None
        vectors = np.load(path, mmap_mode="r", allow_pickle=False)
        if vectors.shape != (expected["input"]["rows"], DIMENSIONS) or vectors.dtype != np.float32:
            return None
        for start in range(0, len(vectors), 8192):
            block = vectors[start:start + 8192]
            if not np.isfinite(block).all() or not np.allclose(
                    np.linalg.norm(block, axis=1), 1.0, atol=1e-4, rtol=1e-5):
                return None
        return vectors, metadata
    except (OSError, ValueError, KeyError, TypeError, EOFError):
        return None


def _save_cache(output: Path, vectors: np.ndarray, metadata: dict) -> dict:
    """Commit the array atomically; metadata is the final completion marker."""
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "metadata.json"
    marker.unlink(missing_ok=True)
    array_temp = metadata_temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=output, prefix=".history-", suffix=".npy",
                                         delete=False) as stream:
            array_temp = Path(stream.name)
            np.save(stream, vectors, allow_pickle=False)
        checksum = _sha256(array_temp)
        os.replace(array_temp, output / "history.npy")
        result = {**metadata, "output_sha256": checksum}
        with tempfile.NamedTemporaryFile(dir=output, prefix=".metadata-", suffix=".json",
                                         mode="w", encoding="utf-8", delete=False) as stream:
            metadata_temp = Path(stream.name)
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(metadata_temp, marker)
        return result
    finally:
        for path in (array_temp, metadata_temp):
            if path is not None:
                path.unlink(missing_ok=True)


def encode_history(history: pd.DataFrame, output: Path, config: dict) -> tuple[np.ndarray, dict]:
    """Encode every ordered own post once, including early unlabelled history.

    Config accepts device (cpu/cuda/mps), batch_size (64), max_length (512),
    bf16 (True; used only on supported CUDA), and show_progress (True). Pass a
    subset and a separate output directory to benchmark before the full run.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    settings = _configuration(config)
    fingerprint = history_fingerprint(history)
    output = Path(output)
    print(json.dumps({"event": "encoder_start", "timestamp": started_at,
                      "model": MODEL, "revision": REVISION,
                      "rows": fingerprint["rows"],
                      "ordered_keys_sha256": fingerprint["ordered_keys_sha256"],
                      "device": settings["device"]}), flush=True)
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install requirements-gpu.txt in the PyTorch environment first") from exc

    device = settings["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    supported_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    use_bf16 = settings["bf16"] and supported_bf16
    compute_dtype = "bfloat16" if use_bf16 else "float32"
    if not settings["bf16"]:
        dtype_reason = "BF16 disabled by config"
    elif device != "cuda":
        dtype_reason = "BF16 is enabled only on supported CUDA devices"
    elif not supported_bf16:
        dtype_reason = "CUDA device does not support BF16; using float32"
    else:
        dtype_reason = "CUDA device supports BF16"
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "sentence-transformers", "transformers", "tokenizers", "numpy", "pandas")}
    versions["python"] = platform.python_version()
    expected = {"schema_version": 1, "model": MODEL, "revision": REVISION,
                "dimensions": DIMENSIONS, "prefix": PREFIX,
                "token_budgets": dict(zip(SOURCE_NAMES[:3], BASE_BUDGETS)),
                "input": fingerprint, "config": settings,
                "runtime": {"compute_dtype": compute_dtype, "dtype_reason": dtype_reason,
                            "versions": versions},
                "encoder_sha256": _sha256(Path(__file__))}
    cached = _load_cache(output, expected)
    if cached is not None:
        print(f"Reusing complete history embeddings: {len(history):,} rows", flush=True)
        return cached

    output.mkdir(parents=True, exist_ok=True)
    (output / "metadata.json").unlink(missing_ok=True)
    load_started = time.monotonic()
    model = SentenceTransformer(MODEL, revision=REVISION, device=device,
                                trust_remote_code=False)
    model.max_seq_length = settings["max_length"]
    model.to(dtype=torch.bfloat16 if use_bf16 else torch.float32)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    load_seconds = time.monotonic() - load_started
    print(f"Encoding on {device} with {compute_dtype}: {dtype_reason}", flush=True)

    records = list(history.loc[:, list(TEXT_COLUMNS)].fillna("").astype(str).itertuples(
        index=False, name=None))
    vectors = np.empty((len(history), DIMENSIONS), dtype=np.float32)
    totals = {"rows": 0, "truncated_rows": 0, "input_tokens": 0, "max_input_tokens": 0,
              "source_tokens": dict.fromkeys(SOURCE_NAMES, 0),
              "kept_tokens": dict.fromkeys(SOURCE_NAMES, 0),
              "truncated_by_source": dict.fromkeys(SOURCE_NAMES, 0)}
    token_seconds = inference_seconds = 0.0
    last_report = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(records), settings["batch_size"]):
            batch = records[start:start + settings["batch_size"]]
            token_started = time.monotonic()
            features, stats = _prepare_input_batch(batch, model.tokenizer, settings["max_length"])
            encoded = model.tokenizer.pad(features, padding=True, return_tensors="pt")
            token_seconds += time.monotonic() - token_started
            inference_started = time.monotonic()
            encoded = {key: value.to(device) for key, value in encoded.items()}
            # Forward preserves our exact token packing; encode() would tokenize again.
            result = model(encoded)["sentence_embedding"].float().cpu().numpy()
            vectors[start:start + len(batch)] = _normalise(result, len(batch))
            inference_seconds += time.monotonic() - inference_started
            for key in ("rows", "truncated_rows", "input_tokens"):
                totals[key] += stats[key]
            totals["max_input_tokens"] = max(totals["max_input_tokens"], stats["max_input_tokens"])
            for key in ("source_tokens", "kept_tokens", "truncated_by_source"):
                for source in SOURCE_NAMES:
                    totals[key][source] += stats[key][source]
            if settings["show_progress"] and time.monotonic() - last_report >= 15:
                print(f"Encoded {start + len(batch):,}/{len(records):,} history posts", flush=True)
                last_report = time.monotonic()
    if history_fingerprint(history) != fingerprint:
        raise ValueError("history changed while encoding; refusing to write a mismatched cache")
    metadata = {**expected, "started_at": started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "output_dtype": "float32", "stats": totals,
                "seconds": {"model_load": round(load_seconds, 3),
                            "tokenize": round(token_seconds, 3),
                            "inference": round(inference_seconds, 3),
                            "total_before_save": round(time.monotonic() - started, 3)}}
    metadata = _save_cache(output, vectors, metadata)
    print(f"Saved {len(vectors):,} history embeddings; truncated {totals['truncated_rows']:,} rows",
          flush=True)
    return vectors, metadata
