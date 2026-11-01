"""Local frozen multilingual embeddings. Does not load any labels or test texts."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

root = Path("results/fusion")
parser = argparse.ArgumentParser()
parser.add_argument("--input", default=str(root / "encoder_input.jsonl"))
parser.add_argument("--prefix", default="")
args = parser.parse_args()

input_path = Path(args.input)
records = [json.loads(line) for line in input_path.open()]
texts = [r["text"] or "" for r in records]

model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
revision = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
metadata_path = root / (args.prefix + "embedding_metadata.json")
vector_path = root / (args.prefix + "multilingual_embeddings.npy")

if metadata_path.exists() and vector_path.exists():
    cached = json.loads(metadata_path.read_text())
    if (
        cached.get("revision") == revision
        and cached.get("input_sha256")
        == hashlib.sha256(input_path.read_bytes()).hexdigest()
    ):
        print("Reusing verified embedding cache:", vector_path, flush=True)
        raise SystemExit(0)

print("Loading", model_id, "revision", revision, flush=True)
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_num_threads(4)
encoder = SentenceTransformer(model_id, revision=revision, device=device)
encoder.eval()

unique = list(dict.fromkeys(t for t in texts if t.strip()))
print(
    "Device",
    device,
    "unique non-empty texts",
    len(unique),
    "max_seq_length",
    encoder.max_seq_length,
    flush=True,
)

start = time.time()
vectors = np.empty((len(unique), 384), dtype=np.float32)
for offset in range(0, len(unique), 1024):
    batch = unique[offset : offset + 1024]
    vectors[offset : offset + len(batch)] = encoder.encode(
        batch,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    done = offset + len(batch)
    elapsed = time.time() - start
    print(
        f"Encoded {done}/{len(unique)}; "
        f"{done / max(elapsed, 0.1):.1f} texts/s; "
        f"remaining {(len(unique) - done) * elapsed / done / 60:.1f} min",
        flush=True,
    )

index = {text: i for i, text in enumerate(unique)}
result = np.zeros((len(texts), 384), dtype=np.float32)
for row, text in enumerate(texts):
    if text.strip():
        result[row] = vectors[index[text]]

assert np.isfinite(result).all()
np.save(vector_path, result)

meta = {
    "model": model_id,
    "revision": revision,
    "dimensions": 384,
    "max_seq_length": encoder.max_seq_length,
    "normalized": True,
    "device": device,
    "rows": len(texts),
    "empty_vectors": "zero",
    "seconds": time.time() - start,
    "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
    "torch_version": torch.__version__,
}
metadata_path.write_text(json.dumps(meta, indent=2))
print("Saved", result.shape, "local embeddings.", flush=True)
