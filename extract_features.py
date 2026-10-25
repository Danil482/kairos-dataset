import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageStat
from sklearn.feature_extraction.text import TfidfVectorizer
from multiprocessing import Pool, cpu_count

DATA_DIR = Path("./data")
MEDIA_DIR = DATA_DIR / "media"
MANIFEST_PATH = MEDIA_DIR / "manifest.jsonl"
OUTPUT_DIR = Path("./extracted_features")
OUTPUT_DIR.mkdir(exist_ok=True)


def process_single_line(line):
    try:
        item = json.loads(line.strip())
    except Exception:
        return None

    urn = item.get("urn") or item.get("post") or item.get("id")
    if not urn:
        return None

    filename = item.get("filename") or item.get("file") or ""
    img_path = os.path.join("./data/media", filename)

    features = [0.0, 0.0, 0.0, 0.0, 0.0]

    if filename and os.path.exists(img_path) and img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                ratio = w / h if h > 0 else 1.0

                small_img = img.resize((32, 32)).convert("L")
                stat = ImageStat.Stat(small_img)

                brightness = stat.mean[0] / 255.0 if stat.mean else 0.5
                rms_contrast = stat.rms[0] / 255.0 if stat.rms else 0.0

                features = [float(w), float(h), float(ratio), float(brightness), float(rms_contrast)]
        except Exception:
            pass

    return urn, features


def extract_fast_image_features_mp():
    print("--- Multithread feature extraction ---")
    if not MANIFEST_PATH.exists():
        print("manifest not found")
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    image_features = {}

    num_workers = max(1, cpu_count() - 2)

    with Pool(processes=num_workers) as pool:
        results = pool.imap_unordered(process_single_line, lines, chunksize=500)

        for res in tqdm(results, total=len(lines), desc="Media files analisys"):
            if res is not None:
                urn, features = res
                image_features[urn] = features

    with open(OUTPUT_DIR / "fast_image_features.json", "w", encoding="utf-8") as f:
        json.dump(image_features, f)
    print("Image features extracted successfully")


def extract_fast_text_features(df: pd.DataFrame, max_features=500):
    print(f"--- Text features extraction (TF-IDF, top-{max_features}) ---")
    texts = df["text"].fillna("").astype(str).tolist()
    posts = df["post"].tolist()

    vectorizer = TfidfVectorizer(max_features=max_features, stop_words=None)
    tfidf_matrix = vectorizer.fit_transform(texts).toarray()

    np.save(OUTPUT_DIR / "fast_text_embeddings.npy", tfidf_matrix)
    with open(OUTPUT_DIR / "fast_text_posts.json", "w", encoding="utf-8") as f:
        json.dump(posts, f)
    print("TF-IDF features saved.")


if __name__ == "__main__":
    from kairos.data import load_posts
    from kairos.label import author_level

    print("Loading data...")
    raw_posts = load_posts(DATA_DIR)
    levelled_posts = author_level(raw_posts)

    extract_fast_text_features(levelled_posts, max_features=500)

    extract_fast_image_features_mp()
