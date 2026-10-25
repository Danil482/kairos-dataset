"""Frozen local encoders with resumable, content-addressed SQLite caches."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import atomic_json, digest, identity, mark, stop_check, verified_marker


class VectorCache:
    def __init__(self, path: Path, fingerprint: str):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (identity TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, data BLOB NOT NULL)")
        record = self.db.execute("SELECT identity FROM metadata").fetchone()
        if record and record[0] != fingerprint:
            self.db.close()
            raise ValueError("Encoder cache identity changed")
        if not record:
            self.db.execute("INSERT INTO metadata VALUES (?)", (fingerprint,))
            self.db.commit()

    def get(self, key: str):
        row = self.db.execute("SELECT data FROM vectors WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        value = np.frombuffer(row[0], dtype=np.float32).copy()
        if not np.isfinite(value).all():
            raise ValueError("Non-finite cached vector")
        return value

    def put(self, key: str, value: np.ndarray):
        self.db.execute("INSERT INTO vectors VALUES (?, ?)", (key, np.asarray(value, dtype=np.float32).tobytes()))

    def commit(self):
        self.db.commit()

    def close(self):
        self.db.close()


def save_array(path: Path, value: np.ndarray):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    tmp.replace(path)


def model_path(models: Path, role: str, settings: dict) -> Path:
    folder = models / role / settings["revision"]
    if not verified_marker(folder / "verified.json", identity(settings)):
        raise ValueError(f"Model snapshot is not verified: {folder}")
    return folder


def encode_text(models: Path, work: Path, config: dict, device: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    settings = config["text_encoder"]
    folder = model_path(models, "text_encoder", settings)
    cache_dir = work / "cache"
    fp = identity({"settings": settings, "data": digest(cache_dir / "all_posts.parquet")})
    marker = cache_dir / "text_embeddings.json"
    if verified_marker(marker, fp):
        return
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable; no automatic device fallback")
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    model = AutoModel.from_pretrained(folder, local_files_only=True).eval().to(device)
    posts = pd.read_parquet(cache_dir / "all_posts.parquet")
    texts = (posts.text.fillna("") + "\n" + posts.document_text.fillna("")).str.strip().tolist()
    keys = [identity(text) for text in texts]
    unique = dict(zip(keys, texts))
    vectors = VectorCache(cache_dir / "text.sqlite", identity(settings))
    width = model.config.hidden_size
    started, completed, truncated, tokens_total = time.monotonic(), 0, 0, 0
    pending = []

    def flush():
        nonlocal completed, truncated, tokens_total
        if not pending:
            return
        inputs, owners = [], []
        encoded = {key: [] for key, _ in pending}
        for key, text in pending:
            tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
            capacity = settings["max_length"] - tokenizer.num_special_tokens_to_add()
            chunks = [tokens[i:i + capacity] for i in range(0, len(tokens), capacity)] or [[]]
            truncated += int(len(chunks) > settings["max_chunks"])
            tokens_total += len(tokens)
            if len(chunks) > settings["max_chunks"]:
                chunks = [chunks[int(i)] for i in np.linspace(0, len(chunks) - 1, settings["max_chunks"])]
            for chunk in chunks:
                inputs.append(tokenizer.prepare_for_model(chunk, add_special_tokens=True, truncation=False))
                owners.append(key)
        for offset in range(0, len(inputs), settings["batch_size"]):
            stop_check(work)
            batch = tokenizer.pad(inputs[offset:offset + settings["batch_size"]], padding=True, return_tensors="pt").to(device)
            with torch.inference_mode():
                output = model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1)
                pooled = ((output * mask).sum(1) / mask.sum(1).clamp(min=1)).float().cpu().numpy()
            for key, value in zip(owners[offset:offset + settings["batch_size"]], pooled):
                encoded[key].append(value)
        for key, text in pending:
            value = np.stack(encoded[key]).mean(0) if text else np.zeros(width, np.float32)
            value /= max(float(np.linalg.norm(value)), 1e-12)
            vectors.put(key, value)
        completed += len(pending)
        vectors.commit()
        pending.clear()
        atomic_json(work / "encoder-status.json", {"stage": "text", "new_texts": completed,
                    "unique_texts": len(unique), "elapsed_seconds": time.monotonic() - started})

    try:
        for key, text in unique.items():
            stop_check(work)
            if vectors.get(key) is not None:
                continue
            pending.append((key, text))
            if len(pending) >= settings["batch_size"]:
                flush()
        flush()
        vectors.commit()
        array = np.stack([vectors.get(key) for key in keys])
        if array.shape != (len(posts), width):
            raise ValueError("Text cache coverage/dimensions mismatch")
        save_array(cache_dir / "text_embeddings.npy", array)
        mark(marker, fp, [cache_dir / "text_embeddings.npy"], rows=len(posts), width=width,
             newly_encoded=completed, new_truncated_texts=truncated, new_tokens=tokens_total,
             chunk_policy="uniform_across_document", elapsed_seconds=time.monotonic() - started)
    finally:
        vectors.close()
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


class MediaSchemaError(ValueError):
    pass


def media_mapping(data: Path, max_images: int) -> tuple[dict, dict]:
    manifest = data / "media/manifest.jsonl"
    if not manifest.exists():
        raise MediaSchemaError("No media manifest in release")
    mapping, keys_seen, matched = {}, set(), 0
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def paths(value):
        if isinstance(value, dict):
            role = str(value.get("role", value.get("type", ""))).lower()
            if any(x in role for x in ("avatar", "logo", "profile")) or value.get("from_reshared_post"):
                return
            sha = value.get("sha256")
            ext = str(value.get("ext", value.get("extension", ""))).lstrip(".")
            if sha and ext:
                yield f"media/{sha[:2]}/{sha}.{ext}"
            for child in value.values():
                yield from paths(child)
        elif isinstance(value, list):
            for child in value:
                yield from paths(child)
        elif isinstance(value, str) and "://" not in value and Path(value).suffix.lower() in extensions:
            yield value

    with manifest.open() as stream:
        for line in stream:
            row = json.loads(line)
            keys_seen.update(row)
            author, post = row.get("author"), row.get("post", row.get("urn"))
            if not isinstance(author, str) or not isinstance(post, str) or row.get("from_reshared_post"):
                continue
            selected = mapping.setdefault((author, post), [])
            for name in paths(row):
                relative = Path(name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise MediaSchemaError("Unsafe path in media manifest")
                candidates = [data / relative, data / "media" / relative]
                for path in candidates:
                    if path.is_file() and path.suffix.lower() in extensions:
                        if not path.resolve().is_relative_to((data / "media").resolve()):
                            raise MediaSchemaError("Media path escapes release")
                        if path not in selected and len(selected) < max_images:
                            selected.append(path)
                            matched += 1
                        break
    if not matched:
        raise MediaSchemaError(f"No post images matched; inspect manifest schema keys: {sorted(keys_seen)}")
    return mapping, {"keys": sorted(keys_seen), "selected_references": matched}


def encode_images(data: Path, models: Path, work: Path, config: dict, device: str):
    import torch
    from PIL import Image, UnidentifiedImageError
    from transformers import CLIPImageProcessor, CLIPModel

    settings = config["image_encoder"]
    folder = model_path(models, "image_encoder", settings)
    cache_dir = work / "cache"
    fp = identity({"settings": settings, "data": digest(cache_dir / "all_posts.parquet")})
    marker = cache_dir / "image_embeddings.json"
    if verified_marker(marker, fp):
        return
    mapping, audit = media_mapping(data, settings["max_images_per_post"])
    torch.set_num_threads(config["threads"])
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    processor = CLIPImageProcessor.from_pretrained(folder, local_files_only=True)
    model = CLIPModel.from_pretrained(folder, local_files_only=True).eval().to(device)
    posts = pd.read_parquet(cache_dir / "all_posts.parquet", columns=["author", "post"])
    vectors = VectorCache(cache_dir / "images.sqlite", identity(settings))
    image_files = sorted({path for key in posts.itertuples(index=False, name=None) for path in mapping.get(key, [])})
    bad, hashes = [], {}
    try:
        pending, images = [], []

        def flush():
            if not images:
                return
            batch = processor(images=images, return_tensors="pt").to(device)
            with torch.inference_mode():
                values = model.get_image_features(**batch)
                values = torch.nn.functional.normalize(values, dim=-1).float().cpu().numpy()
            for key, value in zip(pending, values):
                vectors.put(key, value)
            vectors.commit()
            pending.clear(); images.clear()

        for index, path in enumerate(image_files):
            stop_check(work)
            key = digest(path)
            hashes[path] = key
            if vectors.get(key) is not None or key in pending:
                continue
            try:
                with Image.open(path) as img:
                    images.append(img.convert("RGB"))
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
                bad.append(str(path.relative_to(data)))
                continue
            pending.append(key)
            if len(images) >= settings["batch_size"]:
                flush()
            if index % 100 == 0:
                atomic_json(work / "encoder-status.json", {"stage": "images", "files_seen": index,
                            "files_total": len(image_files), "corrupt": len(bad)})
        flush()
        width = model.config.projection_dim
        result = np.zeros((len(posts), width + 1), dtype=np.float32)
        for i, key in enumerate(posts.itertuples(index=False, name=None)):
            values = [vectors.get(hashes[p]) for p in mapping.get(key, [])]
            values = [x for x in values if x is not None]
            if values:
                result[i, :width] = np.mean(values, axis=0)
                result[i, width] = len(values)
        if not np.isfinite(result).all():
            raise ValueError("Non-finite media embeddings")
        save_array(cache_dir / "image_embeddings.npy", result)
        atomic_json(cache_dir / "media_audit.json", {**audit, "corrupt": bad, "unique_images": len(image_files),
                    "posts_with_images": int((result[:, -1] > 0).sum()), "total_posts": len(posts)})
        mark(marker, fp, [cache_dir / "image_embeddings.npy", cache_dir / "media_audit.json"], rows=len(posts))
    finally:
        vectors.close()
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


def contextual_embeddings(posts: pd.DataFrame, values: np.ndarray, outcomes: np.ndarray | None = None,
                          work: Path | None = None) -> np.ndarray:
    """New-post vector, prior centroid, novelty, and similarity-weighted prior outcomes."""
    if len(posts) != len(values):
        raise ValueError("Embeddings do not match all_posts rows")
    raw = values[:, :]
    width = raw.shape[1]
    output = np.zeros((len(posts), width * 2 + 8), dtype=np.float32)
    output[:, :width] = raw
    for _, group in posts.groupby("author", sort=False):
        if work is not None:
            stop_check(work)
        idx = group.index.to_numpy()
        times = group.posted_at.astype("int64").to_numpy()
        for j, current in enumerate(idx):
            end = int(np.searchsorted(times, times[j], side="left"))
            prior_idx = idx[max(0, end - 20):end]
            if not len(prior_idx):
                continue
            prior = raw[prior_idx]
            centroid = prior.mean(0)
            output[current, width:width * 2] = centroid
            similarities = prior @ raw[current]
            norm = max(float(np.linalg.norm(centroid)), 1e-12)
            output[current, -8:-3] = [float(centroid @ raw[current] / norm), float(similarities.max()),
                                       float(similarities.mean()), float(similarities.std()), len(prior_idx)]
            if outcomes is not None:
                previous = np.log1p(np.maximum(outcomes[prior_idx], 0))
                weights = np.exp(5 * (similarities - similarities.max()))
                weighted = float(np.average(previous, weights=weights))
                output[current, -3:] = [weighted, weighted - float(posts.loc[current, "level"]),
                                        float(previous[int(similarities.argmax())])]
    return output
