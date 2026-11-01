from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from huggingface_hub import model_info
from sentence_transformers import SentenceTransformer

from kairos.data import load_posts
from lab1_features import clean_text
from lab1_metadata import enrich_posts
from train_lab1 import checked_table

MODEL = "intfloat/multilingual-e5-small"


def fingerprint(frame, columns):
    digest = hashlib.sha256()
    for column in columns:
        for value in frame[column]:
            digest.update(str(value).encode())
            digest.update(b"\0")
    return digest.hexdigest()


def history_texts(data, table):
    raw = load_posts(data)
    raw["history_order"] = raw.sort_values(["author", "posted_at"], kind="mergesort").groupby("author").cumcount()
    raw = raw[raw.author.isin(table.author)]
    keys = table[["author", "post"]].assign(is_labelled=True)
    history = raw.merge(keys, on=["author", "post"], how="left")
    history = enrich_posts(history[history.is_labelled.isna()].drop(columns="is_labelled"), data)
    history["clean_text"] = (history.text + " " + history.extra_text.fillna("")).map(clean_text)
    return history


def input_channels(data, table):
    return {
        "posts": (table, ["author", "post"], "clean_text"),
        "profiles": (table[["author", "profile_text"]].drop_duplicates("author").sort_values("author"),
                     ["author"], "profile_text"),
        "history": (history_texts(data, table), ["author", "post", "posted_at", "history_order"], "clean_text"),
    }


def channel_hashes(channels):
    return {name: fingerprint(frame, keys + [column]) for name, (frame, keys, column) in channels.items()}


def main():
    ap = argparse.ArgumentParser(description="Encode post/profile texts locally with multilingual E5")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--output", type=Path, default=Path(".cache/embeddings"))
    ap.add_argument("--reuse-from", type=Path)
    ap.add_argument("--revision")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    torch.set_num_threads(4)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    revision = args.revision or model_info(MODEL, token=False).sha
    table, _ = checked_table(args.data)
    channels = input_channels(args.data, table)
    hashes = channel_hashes(channels)
    metadata = {"model": MODEL, "revision": revision, "prefix": "query: ", "max_length": 512,
                "normalized": True,
                "input_sha256": fingerprint(table, ["author", "post", "clean_text", "profile_text"]),
                "channel_hashes": hashes, "history_input_sha256": hashes["history"],
                "posts": len(table), "authors": len(channels["profiles"][0]),
                "history_posts": len(channels["history"][0]), "device": device,
                "torch": torch.__version__}
    target_info = args.output / "encoder.json"
    if target_info.is_file():
        previous = json.loads(target_info.read_text())
        if previous.get("revision") == revision and previous.get("channel_hashes") == hashes:
            if all((args.output / f"{name}.{ext}").is_file()
                   for name in channels for ext in ["npy", "csv"]):
                print("Matching complete embeddings already exist", flush=True)
                return
        raise SystemExit("Embedding directory contains another or incomplete run; choose another --output")
    previous = None
    if args.reuse_from:
        previous = json.loads((args.reuse_from / "encoder.json").read_text())
        if any(previous.get(k) != metadata[k] for k in ["model", "revision", "prefix", "max_length", "normalized"]):
            raise SystemExit("Reusable vectors have a different encoder configuration")
    encoder = None
    metadata["reused_channels"] = []
    for name, (frame, keys, column) in channels.items():
        output = args.output / f"{name}.npy"
        if previous and previous.get("channel_hashes", {}).get(name) == hashes[name]:
            source = args.reuse_from / f"{name}.npy"
            if len(np.load(source, mmap_mode="r")) != len(frame):
                raise ValueError("Reusable vectors have the wrong number of rows")
            os.link(source, output)
            metadata["reused_channels"].append(name)
            print(f"Reused verified {name} vectors", flush=True)
        else:
            if encoder is None:
                print(f"Encoder {MODEL}@{revision}, device={device}", flush=True)
                encoder = SentenceTransformer(MODEL, revision=revision, device=device, token=False,
                                              trust_remote_code=False, model_kwargs={"use_safetensors": True})
                encoder.max_seq_length = 512
            unique, inverse = np.unique(frame[column].to_numpy(dtype=object), return_inverse=True)
            print(f"Encoding {name}: {len(frame)} rows / {len(unique)} distinct texts", flush=True)
            vectors = encoder.encode(["query: " + s for s in unique], batch_size=args.batch_size,
                                     show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)
            np.save(output, vectors[inverse].astype(np.float32))
        frame[keys].to_csv(args.output / f"{name}.csv", index=False)
    metadata["elapsed_seconds"] = round(time.time() - start, 1)
    target_info.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
