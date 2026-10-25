"""Small SetFit-inspired pairwise adaptation, followed by local encoding.

Uses cosine-similarity MSE on same/different-class pairs. This deliberately uses
the installed Sentence Transformers/PyTorch libraries, not the SetFit package.
The downstream classifier is the existing history-aware gradient booster.
"""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--full",
    action="store_true",
    help="Use the full-train protocol, then encode train and test after selection.",
)
args = parser.parse_args()
root = Path("results/adaptation") / ("full" if args.full else f"seed{args.seed}")
protocol = json.loads((root / "protocol.json").read_text())
pairs = [json.loads(line) for line in (root / "training_pairs.jsonl").open()]
assert len(pairs) == 768
assert sum(p["target"] for p in pairs) == 384
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.set_num_threads(4)
device = "mps" if torch.backends.mps.is_available() else "cpu"
encoder = SentenceTransformer(
    protocol["encoder"],
    revision=protocol["revision"],
    device=device,
    local_files_only=True,
)
assert encoder.max_seq_length == 128
before = (
    next(
        p
        for n, p in encoder.named_parameters()
        if "encoder.layer.0.attention.self.query.weight" in n
    )
    .detach()
    .cpu()
    .clone()
)
optimizer = torch.optim.AdamW(encoder.parameters(), lr=2e-5, weight_decay=0.01)
losses = []
started = time.monotonic()
encoder.train()
for start in range(0, len(pairs), 16):
    batch = pairs[start : start + 16]
    left = batch_to_device(encoder.preprocess([p["left"] for p in batch]), device)
    right = batch_to_device(encoder.preprocess([p["right"] for p in batch]), device)
    target = torch.tensor(
        [p["target"] for p in batch], dtype=torch.float32, device=device
    )
    optimizer.zero_grad(set_to_none=True)
    a = encoder(left)["sentence_embedding"]
    b = encoder(right)["sentence_embedding"]
    loss = torch.nn.functional.mse_loss(
        torch.nn.functional.cosine_similarity(a, b), target
    )
    assert torch.isfinite(loss).item()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
    optimizer.step()
    losses.append(float(loss.detach().cpu()))
    if len(losses) == 1 or len(losses) % 8 == 0:
        print(
            f"Adaptation step {len(losses)}/48; loss={losses[-1]:.4f}; seconds={time.monotonic()-started:.1f}",
            flush=True,
        )
training_seconds = time.monotonic() - started
after = (
    next(
        p
        for n, p in encoder.named_parameters()
        if "encoder.layer.0.attention.self.query.weight" in n
    )
    .detach()
    .cpu()
)
change = float(torch.linalg.vector_norm(after - before))
assert change > 0, "Encoder weights did not change"
del optimizer, left, right, a, b, target, loss
encoder.eval()
encoder.save(str(root / "encoder"), create_model_card=False)
if device == "mps":
    torch.mps.empty_cache()

input_path = Path("results/fusion/encoder_input.jsonl")
records = [json.loads(line) for line in input_path.open()]
train_rows = len(records)
test_input_path = Path("results/fusion/test_encoder_input.jsonl")
if args.full:
    records.extend(json.loads(line) for line in test_input_path.open())
texts = [r["text"] or "" for r in records]
unique = list(dict.fromkeys(t for t in texts if t.strip()))
vectors = np.empty((len(unique), 384), dtype=np.float32)
started = time.monotonic()
for offset in range(0, len(unique), 1024):
    batch = unique[offset : offset + 1024]
    vectors[offset : offset + len(batch)] = encoder.encode(
        batch,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    if offset % 8192 == 0 or offset + len(batch) == len(unique):
        print(
            f"Adapted encoding: {offset+len(batch)}/{len(unique)}; seconds={time.monotonic()-started:.1f}",
            flush=True,
        )
index = {text: i for i, text in enumerate(unique)}
result = np.zeros((len(texts), 384), dtype=np.float32)
for i, text in enumerate(texts):
    if text.strip():
        result[i] = vectors[index[text]]
assert np.isfinite(result).all()
np.save(root / "embeddings.npy", result[:train_rows])
if args.full:
    np.save(root / "test_embeddings.npy", result[train_rows:])
metadata = {
    "method": "SetFit-inspired cosine-MSE pairwise encoder fine-tuning, not the SetFit package",
    "protocol": protocol,
    "device": device,
    "training_seconds": training_seconds,
    "encoding_seconds": time.monotonic() - started,
    "steps": len(losses),
    "losses": losses,
    "checked_layer_weight_change_l2": change,
    "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
    "pairs_sha256": hashlib.sha256(
        (root / "training_pairs.jsonl").read_bytes()
    ).hexdigest(),
    "rows": train_rows,
    "dimensions": 384,
    "torch": torch.__version__,
}
if args.full:
    metadata["test_rows"] = len(result) - train_rows
    metadata["test_input_sha256"] = hashlib.sha256(
        test_input_path.read_bytes()
    ).hexdigest()
(root / "encoder_training.json").write_text(json.dumps(metadata, indent=2))
print("Adapted embeddings saved; encoder weight change verified.", flush=True)
