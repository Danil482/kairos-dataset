"""Local SigLIP 2 content-image vectors for official train posts only."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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
import pandas as pd

from scripts.semantic_embeddings import SOURCE, fingerprint, save_json

OUT = ROOT / "results" / "visual"
MEDIA = ROOT / "data" / "media"
MODEL_ID = "google/siglip2-base-patch16-224"
KINDS = ("post_image", "article_image", "video_poster", "document_cover", "document_page", "celebration_image")


def media_path(relative, root=MEDIA):
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Media path leaves the dataset directory")
    return path


def select_content_rows(rows, keys):
    """Keep one post/file pair; repeated rows cannot increase its image weight."""
    pairs = {}
    excluded = Counter()
    for row in rows:
        key = (row.get("author"), row.get("post"))
        if key not in keys:
            continue
        if row.get("kind") not in KINDS or not (row.get("content_type") or "").startswith("image/"):
            excluded[str(row.get("kind"))] += 1
            continue
        file = row.get("file")
        if not isinstance(file, str) or not file:
            raise ValueError("Content image row has no local path")
        pair = pairs.setdefault((*key, file), dict(author=key[0], post=key[1], file=file, kinds=set(), own=False, shared=False))
        pair["kinds"].add(row["kind"])
        pair["shared"] |= bool(row.get("from_reshared_post"))
        pair["own"] |= not bool(row.get("from_reshared_post"))
    return [{**pair, "kinds":sorted(pair["kinds"])} for pair in pairs.values()], dict(excluded)


def build_index():
    table = joblib.load(SOURCE)["table"]
    if set(table.split) != {"train"}:
        raise ValueError("Only official train posts may enter this experiment")
    keys = set(zip(table.author, table.post))
    with (MEDIA / "manifest.jsonl").open(encoding="utf-8") as stream:
        pairs, excluded = select_content_rows((json.loads(line) for line in stream), keys)
    files = sorted({pair["file"] for pair in pairs})
    missing = [name for name in files if not media_path(name).is_file()]
    if missing:
        raise RuntimeError(f"{len(missing)} content image files are missing; finish archive preparation")
    index = {name:i for i,name in enumerate(files)}
    for pair in pairs:
        pair["image_index"] = index[pair["file"]]
    config = dict(table_sha256=fingerprint(SOURCE), media_manifest_sha256=fingerprint(MEDIA / "manifest.jsonl"),
                  kinds=list(KINDS), pairing="unique author/post/file", format_version=1)
    path = OUT / "media_index.json"
    if path.exists() and json.loads(path.read_text())["config"] != config:
        raise ValueError("Existing visual index uses a different dataset")
    save_json(path, dict(config=config, files=files, pairs=pairs, excluded_rows=excluded))
    table[["author", "post"]].to_csv(OUT / "train_keys.csv", index=False)
    print(f"Indexed {len(files)} unique content images, {len(pairs)} post/file pairs, {len({(p['author'],p['post']) for p in pairs})} posts", flush=True)


def download_encoder():
    from huggingface_hub import HfApi, snapshot_download
    manifest = OUT / "encoder.json"
    if manifest.exists():
        spec = json.loads(manifest.read_text())
        if all((Path(spec["local_path"]) / name).is_file() for name in ("config.json", "preprocessor_config.json", "model.safetensors")):
            print("Pinned visual encoder is already available locally", flush=True)
            return
        revision = spec["revision"]
    else:
        revision = HfApi().model_info(MODEL_ID).sha
    path = snapshot_download(MODEL_ID, revision=revision,
        local_dir=ROOT / ".tools" / "models" / "siglip2-base-patch16-224",
        allow_patterns=["config.json", "preprocessor_config.json", "model*.safetensors", "model.safetensors.index.json"], max_workers=2)
    save_json(manifest, dict(model_id=MODEL_ID, revision=revision, local_path=path,
                            source="https://huggingface.co/" + MODEL_ID))
    print(f"Visual encoder pinned to {revision}", flush=True)


def encode(batch_size):
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageOps, UnidentifiedImageError
    from transformers import AutoImageProcessor, AutoModel

    index = json.loads((OUT / "media_index.json").read_text())
    spec = json.loads((OUT / "encoder.json").read_text())
    config = dict(index=index["config"], model_id=MODEL_ID, revision=spec["revision"], normalization="L2 image features", format_version=2)
    manifest_path = OUT / "embedding_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["config"] != config:
            raise ValueError("Completed visual cache differs from requested configuration")
        print("Visual embedding cache is complete", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this configured visual experiment")
    torch.set_num_threads(4)
    processor = AutoImageProcessor.from_pretrained(spec["local_path"], local_files_only=True, use_fast=False, trust_remote_code=False)
    model = AutoModel.from_pretrained(spec["local_path"], local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=torch.float16, attn_implementation="sdpa").to("cuda").eval()
    files = index["files"]
    progress_path = OUT / "embedding_progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else dict(config=config, completed_images=0, errors=[])
    if progress["config"] != config:
        raise ValueError("Partial visual cache uses a different configuration")
    vectors_path, details_path = OUT / "image_vectors.npy", OUT / "image_details.npy"
    vectors = np.lib.format.open_memmap(vectors_path, mode="r+" if vectors_path.exists() else "w+", dtype=np.float32, shape=(len(files), 768))
    # width, height, on-disk bytes, successfully decoded flag.
    details = np.lib.format.open_memmap(details_path, mode="r+" if details_path.exists() else "w+", dtype=np.float32, shape=(len(files), 4))

    def prepare_image(number):
        try:
            path = media_path(files[number])
            with Image.open(path) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
                width, height = image.size
                pixels = processor(images=image, return_tensors="np", input_data_format="channels_last")["pixel_values"][0]
                image.close()
            return number, pixels, [width, height, path.stat().st_size, 1.], None
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as error:
            return number, None, [0., 0., 0., 0.], type(error).__name__

    start = progress["completed_images"]
    errors = progress["errors"]
    started, last_print = time.monotonic(), 0.
    print(f"Encoding {len(files)} content images on {torch.cuda.get_device_name(0)}; resuming at {start}", flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for offset in range(start, len(files), batch_size):
            stop = min(offset+batch_size, len(files))
            prepared = list(pool.map(prepare_image, range(offset, stop)))
            usable = [item for item in prepared if item[1] is not None]
            vectors[offset:stop] = 0.
            for number, _, information, error in prepared:
                details[number] = information
                if error:
                    errors.append(dict(image_index=number, error_type=error))
            if usable:
                inputs = torch.as_tensor(np.stack([item[1] for item in usable]), dtype=torch.float16, device="cuda")
                with torch.inference_mode():
                    values = F.normalize(model.get_image_features(pixel_values=inputs).float(), p=2, dim=1).cpu().numpy()
                if values.shape[1] != 768 or not np.isfinite(values).all():
                    raise ValueError("Invalid visual feature vectors")
                vectors[[item[0] for item in usable]] = values
            if (stop-start) % (batch_size*10) == 0 or stop == len(files):
                vectors.flush()
                details.flush()
                save_json(progress_path, dict(config=config, completed_images=stop, total_images=len(files), errors=errors))
            if time.monotonic()-last_print >= 30 or stop == len(files):
                elapsed = time.monotonic()-started
                print(f"Visual images {stop}/{len(files)} ({100*stop/len(files):.1f}%), {(stop-start)/max(elapsed,1):.1f} images/s, failures {len(errors)}", flush=True)
                last_print = time.monotonic()
    save_json(manifest_path, dict(config=config, files=len(files), successful_images=int(details[:,3].sum()),
        errors=errors, dimensions=768, gpu=torch.cuda.get_device_name(0), seconds=round(time.monotonic()-started,1), complete=True))
    print("Visual image vectors complete", flush=True)


def aggregate_posts(table, index, vectors, details):
    """Average unique successfully decoded content images; keep missing posts."""
    lookup = {(row.author, row.post):i for i,row in enumerate(table.itertuples())}
    mean = np.zeros((len(table), vectors.shape[1]), dtype=np.float32)
    first = np.zeros_like(mean)
    columns = ["visual_count", "visual_decoded_count", "visual_shared_count", "visual_own_count",
               "visual_log_width", "visual_log_height", "visual_log_bytes", "visual_log_aspect"] + ["visual_"+kind for kind in KINDS]
    meta = np.zeros((len(table), len(columns)), dtype=np.float32)
    for pair in index["pairs"]:
        row = lookup[(pair["author"], pair["post"])]
        number = pair["image_index"]
        meta[row, 0] += 1
        meta[row, 2] += pair["shared"]
        meta[row, 3] += pair["own"]
        for kind in pair["kinds"]:
            meta[row, 8+KINDS.index(kind)] += 1
        width, height, size, success = details[number]
        if success:
            if meta[row, 1] == 0:
                first[row] = vectors[number]
            mean[row] += vectors[number]
            meta[row, 1] += 1
            meta[row, 4:8] += [np.log1p(width), np.log1p(height), np.log1p(size), np.log(width/height)]
    meta[:, 4:8] /= np.maximum(meta[:, 1:2], 1)
    mean /= np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)
    metadata = pd.DataFrame(meta, columns=columns)
    metadata["visual_available"] = (meta[:, 1] > 0).astype(np.float32)
    return mean, first, metadata


def aggregate():
    manifest = json.loads((OUT / "embedding_manifest.json").read_text())
    index = json.loads((OUT / "media_index.json").read_text())
    if not manifest["complete"] or manifest["config"]["index"] != index["config"]:
        raise ValueError("Visual image cache is incomplete or stale")
    table = joblib.load(SOURCE)["table"].reset_index(drop=True)
    if fingerprint(SOURCE) != index["config"]["table_sha256"]:
        raise ValueError("Visual index and training table differ")
    if not table[["author", "post"]].equals(pd.read_csv(OUT / "train_keys.csv")):
        raise ValueError("Visual index and training row order differ")
    mean, first, metadata = aggregate_posts(table, index, np.load(OUT / "image_vectors.npy", mmap_mode="r"),
                                           np.load(OUT / "image_details.npy", mmap_mode="r"))
    np.save(OUT / "post_mean.npy", mean)
    np.save(OUT / "post_first.npy", first)
    joblib.dump(metadata, OUT / "metadata.joblib", compress=3)
    save_json(OUT / "aggregation.json", dict(config=manifest["config"], rows=len(table),
        posts_with_images=int(metadata.visual_count.gt(0).sum()), posts_with_vectors=int(metadata.visual_available.sum()),
        aggregation="L2-normalized mean and first available image in manifest order; unique post/file pairs",
        metadata_columns=metadata.columns.tolist(), complete=True))
    print(f"Aggregated images for {int(metadata.visual_available.sum())}/{len(table)} posts", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.download_only:
        download_encoder()
        return
    build_index()
    encode(args.batch_size)
    aggregate()


if __name__ == "__main__":
    main()
