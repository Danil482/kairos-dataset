"""Local English/Russian OCR of all content images on empty-original train posts."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("EASYOCR_MODULE_PATH",str(ROOT / ".tools" / "easyocr"))
import joblib
import numpy as np
import pandas as pd
from scripts.semantic_embeddings import SOURCE,fingerprint,save_json
from scripts.semantic_history import encode_missing
from scripts.visual_embeddings import media_path

OUT = ROOT / "results" / "empty_text"
OCR = OUT / "ocr"
POLICY = dict(version=1,languages=["ru","en"],canvas_size=1600,batch_size=32,min_confidence=.35,
              min_letters=3,scope="all allowed content images of official train posts with original_chars=0",
              sanitation="drop engagement UI fragments; remove URLs, emails, handles and numeric values")
UI = re.compile(r"\b(?:likes?|comments?|reactions?|reposts?|impressions?|views?|shares?|followers?)\b|нравится|комментар|реакци|репост|просмотр|подписчик",re.I)
LINK = re.compile(r"https?://\S+|www\.\S+|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|(?<!\w)@[\w.]+",re.I)
NUMBER = re.compile(r"\b\d[\d.,]*(?:[kKmM%])?\b")


def clean_fragment(text):
    if UI.search(text):
        return ""
    text = LINK.sub(" ",text)
    text = NUMBER.sub(" ",text)
    text = re.sub(r"\s+"," ",text).strip()
    return text if sum(c.isalpha() for c in text)>=POLICY["min_letters"] else ""


def reader(download=False):
    import torch
    import easyocr
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("This OCR run is configured for CUDA")
    base = ROOT / ".tools" / "easyocr"
    return easyocr.Reader(POLICY["languages"],gpu=True,model_storage_directory=str(base / "models"),
        user_network_directory=str(base / "user_network"),download_enabled=download,verbose=False)


def download_weights():
    from huggingface_hub import hf_hub_download
    from easyocr.config import detection_models,recognition_models
    base = ROOT / ".tools" / "easyocr" / "models"
    base.mkdir(parents=True,exist_ok=True)
    repository = "xiaoyao9184/easyocr"
    revision = "7fcca5cbab94e60f54b1c9c930f162cc24bf43b6"
    records = []
    # The upstream GitHub release host failed DNS here. Verify the mirror's bytes
    # against checksums distributed in the official EasyOCR package before loading.
    for spec in (detection_models["craft"],recognition_models["gen2"]["cyrillic_g2"]):
        path = base / spec["filename"]
        if not path.is_file():
            path = Path(hf_hub_download(repository,spec["filename"],revision=revision,local_dir=base))
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream,"md5").hexdigest()
        if digest != spec["md5sum"]:
            raise ValueError("OCR weights do not match upstream checksum")
        records.append(dict(filename=path.name,upstream_md5=digest,sha256=fingerprint(path)))
    save_json(base / "weights_source.json",dict(mirror=repository,revision=revision,files=records,
        authoritative_checksums="EasyOCR 1.7.2 easyocr/config.py"))
    reader(download=False)
    print("OCR weights verified and loaded locally",flush=True)


def prepare_index():
    OCR.mkdir(parents=True,exist_ok=True)
    table = joblib.load(SOURCE)["table"].reset_index(drop=True)
    if set(table.split)!={"train"}:
        raise ValueError("OCR experiment is train-only")
    source = ROOT / "results" / "visual" / "media_index.json"
    index = json.loads(source.read_text())
    if index["config"]["table_sha256"] != fingerprint(SOURCE):
        raise ValueError("Media index and training table differ")
    keys = set(zip(table.loc[table.original_chars.eq(0),"author"],table.loc[table.original_chars.eq(0),"post"]))
    pairs = [pair for pair in index["pairs"] if (pair["author"],pair["post"]) in keys]
    files = sorted({pair["file"] for pair in pairs})
    config = dict(policy=POLICY,table_sha256=fingerprint(SOURCE),media_index_sha256=fingerprint(source))
    path = OCR / "index.json"
    if path.exists() and json.loads(path.read_text())["config"]!=config:
        raise ValueError("Incompatible existing OCR index")
    save_json(path,dict(config=config,files=files,pairs=pairs))
    return table,files,pairs,config


def extract(limit=None):
    from PIL import Image,ImageOps
    table,files,pairs,config = prepare_index()
    manifest_path = OCR / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing["config"]!=config or existing["posts_sha256"]!=fingerprint(OCR / "posts.joblib"):
            raise ValueError("Completed OCR cache is incompatible or damaged")
        print("OCR text cache already complete",flush=True)
        return
    cache = OCR / "files"
    cache.mkdir(exist_ok=True)
    engine = reader()
    configuration_hash = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    started,last_print,completed = time.monotonic(),0.,0
    for i,file in enumerate(files[:limit]):
        target = cache / f"{i:05d}.json"
        if target.exists():
            saved = json.loads(target.read_text())
            if saved["config_hash"]!=configuration_hash or saved["file"]!=file:
                raise ValueError("OCR file cache does not match")
            completed += 1
            continue
        path = media_path(file)
        result = dict(file=file,config_hash=configuration_hash,sha256=fingerprint(path),fragments=[],error=None)
        try:
            with Image.open(path) as original:
                rgba = ImageOps.exif_transpose(original).convert("RGBA")
                white = Image.new("RGBA",rgba.size,"white")
                pixels = np.asarray(Image.alpha_composite(white,rgba).convert("RGB"))
            if min(pixels.shape[:2])>=8:
                recognized = engine.readtext(pixels,detail=1,paragraph=False,batch_size=POLICY["batch_size"],
                    workers=0,canvas_size=POLICY["canvas_size"],decoder="greedy")
                seen = set()
                for _,text,confidence in recognized:
                    cleaned = clean_fragment(text)
                    if confidence>=POLICY["min_confidence"] and cleaned and cleaned.casefold() not in seen:
                        result["fragments"].append(dict(text=cleaned,confidence=float(confidence)))
                        seen.add(cleaned.casefold())
        except (OSError,ValueError) as error:
            result["error"] = type(error).__name__
        save_json(target,result)
        completed += 1
        if time.monotonic()-last_print>30 or i==len(files)-1:
            print(f"OCR images {completed}/{len(files)}, elapsed {time.monotonic()-started:.0f}s",flush=True)
            save_json(OCR / "progress.json",dict(completed=completed,total=len(files),config_hash=configuration_hash))
            last_print = time.monotonic()
    if completed<len(files):
        print(f"Pilot complete: {completed}/{len(files)} images cached",flush=True)
        return
    saved_files = {file:json.loads((cache / f"{i:05d}.json").read_text()) for i,file in enumerate(files)}
    fragments = {}
    for pair in pairs:
        fragments.setdefault((pair["author"],pair["post"]),[]).append(saved_files[pair["file"]])
    rows = []
    for row in table.itertuples():
        found = fragments.get((row.author,row.post),[])
        text,confidences,seen = [],[],set()
        for item in found:
            for part in item["fragments"]:
                if part["text"].casefold() not in seen:
                    text.append(part["text"])
                    confidences.append(part["confidence"])
                    seen.add(part["text"].casefold())
        joined = "\n".join(text)[:20000]
        tokens = set(re.findall(r"\w+",joined.casefold()))
        context = set(re.findall(r"\w+",row.expanded_text.casefold()))
        rows.append(dict(author=row.author,post=row.post,ocr_text=joined,ocr_available=int(bool(joined)),
            ocr_log_chars=float(np.log1p(len(joined))),ocr_log_fragments=float(np.log1p(len(text))),
            ocr_mean_confidence=float(np.mean(confidences)) if confidences else 0.,
            ocr_images=len(found),ocr_images_with_text=sum(bool(item["fragments"]) for item in found),
            ocr_new_token_share=len(tokens-context)/max(len(tokens),1)))
    data = pd.DataFrame(rows)
    joblib.dump(data,OCR / "posts.joblib",compress=3)
    model_files = {p.name:fingerprint(p) for p in (ROOT / ".tools" / "easyocr" / "models").glob("*.pth")}
    save_json(OCR / "manifest.json",dict(config=config,images=len(files),images_with_text=sum(bool(item["fragments"]) for item in saved_files.values()),
        errors=sum(item["error"] is not None for item in saved_files.values()),posts_with_text=int(data.ocr_available.sum()),
        posts_sha256=fingerprint(OCR / "posts.joblib"),model_sha256=model_files,complete=True))
    print(f"OCR complete: {int(data.ocr_available.sum())} posts with recognized text",flush=True)


def encode():
    manifest = json.loads((OCR / "manifest.json").read_text())
    path = OCR / "embedding_manifest.json"
    if path.exists():
        if json.loads(path.read_text())["posts_sha256"]!=manifest["posts_sha256"]:
            raise ValueError("OCR embeddings refer to another OCR corpus")
        print("OCR E5 cache already complete",flush=True)
        return
    data = joblib.load(OCR / "posts.joblib")
    texts = list(dict.fromkeys(text for text in data.ocr_text if text.strip()))
    lookup = {text:i for i,text in enumerate(texts)}
    unique = encode_missing(texts,cache_dir=OCR / "e5") if texts else np.zeros((0,768),dtype=np.float32)
    vectors = np.zeros((len(data),768),dtype=np.float32)
    for i,text in enumerate(data.ocr_text):
        if text.strip():
            vectors[i] = unique[lookup[text]]
    np.save(OCR / "vectors.npy",vectors)
    save_json(path,dict(posts_sha256=manifest["posts_sha256"],rows=len(data),unique_texts=len(texts),
        encoder=json.loads((ROOT / "results" / "semantic" / "encoder.json").read_text()),complete=True))
    print("OCR text vectors complete",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only",action="store_true")
    parser.add_argument("--limit",type=int)
    parser.add_argument("--encode-only",action="store_true")
    args = parser.parse_args()
    if args.download_only:
        download_weights()
    elif args.encode_only:
        encode()
    else:
        extract(args.limit)
        if args.limit is None:
            encode()


if __name__ == "__main__":
    main()
