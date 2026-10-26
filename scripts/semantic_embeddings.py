"""Cache frozen multilingual E5 vectors locally; no dataset text is sent online."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".tools" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import joblib
import numpy as np

SOURCE = ROOT / "results" / "exploration_v2" / "train_table.joblib"
OUT = ROOT / "results" / "semantic"
MODEL_ID = "intfloat/multilingual-e5-base"
CHUNK_TOKENS = 480
MAX_CHUNKS = 3


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def fingerprint(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def chunk_token_ids(ids, width=CHUNK_TOKENS, limit=MAX_CHUNKS):
    """Cover short texts fully; sample start/middle/end for very long texts."""
    chunks = [ids[i:i+width] for i in range(0, len(ids), width)] or [[]]
    if len(chunks) > limit:
        indices = np.linspace(0, len(chunks)-1, limit).round().astype(int)
        chunks = [chunks[i] for i in indices]
    return chunks


def collect_texts(table):
    unique, lookup, columns = [], {}, {}
    clipped = 0
    for field in ("text", "expanded_text"):
        indices = []
        for value in table[field].fillna("").astype(str):
            clipped += int(len(value) > 20000)
            text = value[:20000].strip()
            if not text:
                indices.append(-1)
                continue
            if text not in lookup:
                lookup[text] = len(unique)
                unique.append(text)
            indices.append(lookup[text])
        columns[field] = np.asarray(indices, dtype=np.int32)
    return unique, columns, clipped


def download_encoder():
    from huggingface_hub import HfApi, snapshot_download
    manifest = OUT / "encoder.json"
    if manifest.exists():
        spec = json.loads(manifest.read_text())
        local = Path(spec["local_path"])
        if all((local / name).is_file() for name in ("config.json", "tokenizer.json", "model.safetensors")):
            print(f"Pinned encoder is already available locally: {spec['revision']}", flush=True)
            return
        revision = spec["revision"]
    else:
        revision = HfApi().model_info(MODEL_ID).sha
    path = snapshot_download(MODEL_ID, revision=revision,
        local_dir=ROOT / ".tools" / "models" / "multilingual-e5-base",
        allow_patterns=["*.json", "*.txt", "*.model", "model.safetensors"],
        ignore_patterns=["onnx/*", "openvino/*"], max_workers=3)
    spec = dict(model_id=MODEL_ID, revision=revision, local_path=path,
                source="https://huggingface.co/" + MODEL_ID)
    save_json(manifest, spec)
    print(f"Encoder downloaded and pinned to {revision}", flush=True)


def encode(batch_size):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; inspect the installation before starting the full corpus")
    torch.set_num_threads(6)
    source_hash = fingerprint(SOURCE)
    spec = json.loads((OUT / "encoder.json").read_text())
    config = dict(input_sha256=source_hash, encoder=spec["model_id"], revision=spec["revision"],
                  chunk_tokens=CHUNK_TOKENS, max_chunks=MAX_CHUNKS, prefix="query: ",
                  normalization="mean pooling with attention mask, then L2; token-weighted chunk average, then L2",
                  max_characters=20000, format_version=1)
    complete = OUT / "embedding_manifest.json"
    if complete.exists():
        old = json.loads(complete.read_text())
        if old["config"] != config:
            raise ValueError("Completed embedding cache has a different input/configuration")
        print("Semantic embedding cache is already complete", flush=True)
        return
    table = joblib.load(SOURCE)["table"]
    if set(table.split) != {"train"} or table.duplicated(["author", "post"]).any():
        raise ValueError("Expected unique official train posts only")
    texts, mappings, clipped = collect_texts(table)
    print(f"Tokenizing {len(texts)} unique local texts from {len(table)} train posts", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(spec["local_path"], local_files_only=True, trust_remote_code=False)
    # Chunk construction below enforces the model limit; avoid a warning during raw tokenization.
    tokenizer.model_max_length = 10**9
    tokens = tokenizer(texts, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)["input_ids"]
    del texts
    prefix = tokenizer("query: ", add_special_tokens=False)["input_ids"]
    records, owners, weights = [], [], []
    sampled_long = 0
    for index, ids in enumerate(tokens):
        sampled_long += int(len(ids) > CHUNK_TOKENS * MAX_CHUNKS)
        for chunk in chunk_token_ids(ids):
            prepared = tokenizer.build_inputs_with_special_tokens(prefix + chunk)
            if len(prepared) > 512:
                raise ValueError("Chunk exceeds E5 context limit")
            records.append(dict(input_ids=prepared, attention_mask=[1]*len(prepared)))
            owners.append(index)
            weights.append(max(len(chunk), 1))
    del tokens
    order = np.argsort([len(r["input_ids"]) for r in records], kind="stable")
    progress_file = OUT / "embedding_progress.json"
    progress = json.loads(progress_file.read_text()) if progress_file.exists() else dict(config=config, completed_chunks=0)
    if progress["config"] != config:
        raise ValueError("Partial cache has a different input/configuration")
    chunk_path = OUT / "chunk_vectors.npy"
    if chunk_path.exists():
        chunk_vectors = np.lib.format.open_memmap(chunk_path, mode="r+")
        if chunk_vectors.shape != (len(records), 768):
            raise ValueError("Chunk cache shape changed")
    else:
        chunk_vectors = np.lib.format.open_memmap(chunk_path, mode="w+", dtype=np.float16, shape=(len(records), 768))
    model = AutoModel.from_pretrained(spec["local_path"], local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=torch.float16, attn_implementation="sdpa").to("cuda").eval()
    started, last_print = time.monotonic(), 0.
    initial = progress["completed_chunks"]
    print(f"Encoding {len(records)} chunks on {torch.cuda.get_device_name(0)}; resuming at {initial}", flush=True)
    for offset in range(initial, len(records), batch_size):
        chosen = order[offset:offset+batch_size]
        batch = tokenizer.pad([records[i] for i in chosen], padding=True, pad_to_multiple_of=8, return_tensors="pt")
        batch = {key:value.to("cuda") for key,value in batch.items()}
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state.float()
            mask = batch["attention_mask"].unsqueeze(-1).bool()
            pooled = hidden.masked_fill(~mask, 0.).sum(1) / mask.sum(1)
            vectors = F.normalize(pooled, p=2, dim=1).cpu().numpy()
        if not np.isfinite(vectors).all():
            raise ValueError("Non-finite E5 vectors")
        chunk_vectors[chosen] = vectors.astype(np.float16)
        stop = min(offset+batch_size, len(records))
        # Checkpoint at deterministic chunk positions; replay after interruption overwrites, never adds twice.
        if stop % (batch_size*20) == 0 or stop == len(records):
            chunk_vectors.flush()
            save_json(progress_file, dict(config=config, completed_chunks=stop, total_chunks=len(records)))
        if time.monotonic()-last_print >= 30 or stop == len(records):
            elapsed = time.monotonic()-started
            print(f"E5 chunks {stop}/{len(records)} ({100*stop/len(records):.1f}%), {(stop-initial)/max(elapsed,1):.1f} chunks/s", flush=True)
            last_print = time.monotonic()
    unique_vectors = np.zeros((len(owners) and max(owners)+1, 768), dtype=np.float32)
    for start in range(0, len(records), 2048):
        end = min(start+2048, len(records))
        np.add.at(unique_vectors, owners[start:end], np.asarray(chunk_vectors[start:end], dtype=np.float32) * np.asarray(weights[start:end], dtype=np.float32)[:, None])
    norms = np.linalg.norm(unique_vectors, axis=1, keepdims=True)
    unique_vectors /= np.maximum(norms, 1e-12)
    np.save(OUT / "unique_vectors.npy", unique_vectors)
    np.savez(OUT / "post_vector_indices.npz", **mappings)
    table[["author", "post"]].to_csv(OUT / "train_keys.csv", index=False)
    save_json(complete, dict(config=config, rows=len(table), unique_texts=len(unique_vectors), chunks=len(records),
        sampled_long_texts=sampled_long, character_clipped_inputs=clipped,
        empty_original=int((mappings["text"] == -1).sum()), empty_expanded=int((mappings["expanded_text"] == -1).sum()),
        dimensions=768, gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
        session_seconds=round(time.monotonic()-started, 1), batch_size=batch_size, complete=True))
    print("Semantic embeddings complete and saved locally", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.download_only:
        download_encoder()
    else:
        encode(args.batch_size)


if __name__ == "__main__":
    main()
