import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from kairos.label import MIN_HISTORY
from PIL import Image


URL_RE = re.compile(r"https?://|⟨URL⟩")
MENTION_RE = re.compile(r"@\w+|⟨HANDLE⟩|⟨(?:PERSON|AUTHOR)[^⟩]*⟩")
LINK_RE = re.compile(r"⟨LINK[^⟩]*⟩")
LINK_OR_URL_RE = re.compile(r"https?://|⟨URL⟩|⟨LINK[^⟩]*⟩")
HASHTAG_RE = re.compile(r"#(?:\w+|⟨[^⟩]*⟩)")
ORG_RE = re.compile(r"⟨ORG[^⟩]*⟩")
PERSON_RE = re.compile(r"⟨PERSON[^⟩]*⟩")
EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF]+"
)

def prev_median(s):
    return s.shift(1).expanding(min_periods=MIN_HISTORY).median()

def prev_mean(s):
    return s.shift(1).expanding(min_periods=MIN_HISTORY).mean()

def prev_std(s):
    return s.shift(1).expanding(min_periods=MIN_HISTORY).std()

def prev_min(s):
    return s.shift(1).expanding(min_periods=MIN_HISTORY).min()

def prev_max(s):
    return s.shift(1).expanding(min_periods=MIN_HISTORY).max()

def prev_q25(s):
    return s.shift(1).expanding(min_periods=MIN_HISTORY).quantile(0.25)

def prev_q75(s):
    return s.shift(1).expanding(min_periods=MIN_HISTORY).quantile(0.75)

def load_media_counts(data_dir):
    path = Path(data_dir) / "media" / "manifest.jsonl"
    empty = pd.DataFrame(columns=["author", "post", "n_images", "n_videos", "n_pdfs"])
    if not path.exists():
        print("no manifest.jsonl, media features = 0")
        return empty

    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        return empty

    m = pd.DataFrame(rows)
    if "author" not in m.columns or "post" not in m.columns:
        return empty
    if "from_reshared_post" in m.columns:
        m = m[~m["from_reshared_post"].fillna(False).astype(bool)]

    if "type" in m.columns:
        kind = m["type"]
    else:
        file_col = next((c for c in ("file", "path", "filename") if c in m.columns), None)
        if file_col is None:
            return empty
        ext = m[file_col].astype(str).str.rsplit(".", n=1).str[-1].str.lower()
        kind = np.where(
            ext.isin(["jpg", "jpeg", "png", "gif", "webp"]), "image",
            np.where(
                ext.isin(["mp4", "mov", "avi", "webm", "mkv"]), "video",
                np.where(ext.eq("pdf"), "pdf", "other"),
            ),
        )

    m = m.assign(kind=kind)
    agg = (m.groupby(["author", "post", "kind"]).size().unstack(fill_value=0).reset_index().rename(columns={"image": "n_images", "video": "n_videos", "pdf": "n_pdfs"}))
    for col in ("n_images", "n_videos", "n_pdfs"):
        if col not in agg.columns:
            agg[col] = 0
    return agg[["author", "post", "n_images", "n_videos", "n_pdfs"]]


def build_dense_features(df, followers, media):
    df = df.copy().sort_values(["author", "posted_at"]).reset_index(drop=True)

    text = df["text"].fillna("")
    df["n_chars"] = text.str.len()
    df["n_words"] = text.str.split().str.len().fillna(0)
    df["n_lines"] = text.str.count("\n") + 1
    df["n_hashtags"] = text.str.count(HASHTAG_RE)
    df["n_mentions"] = text.str.count(MENTION_RE)
    df["n_urls"] = text.str.count(URL_RE)
    df["n_orgs"] = text.str.count(ORG_RE)
    df["n_people"] = text.str.count(PERSON_RE)
    df["n_emoji"] = text.str.count(EMOJI_RE)
    df["has_question"] = text.str.contains("?", regex=False).astype(int)
    df["has_exclaim"] = text.str.contains("!", regex=False).astype(int)
    df["is_textless"] = (df["n_chars"] == 0).astype(int)
    df["hour"] = df["posted_at"].dt.hour.fillna(-1).astype(int)
    df["weekday"] = df["posted_at"].dt.weekday.fillna(-1).astype(int)

    df["log_followers"] = np.log1p(df["author"].map(followers).fillna(0).clip(lower=0))
    age = (df["crawled_at"] - df["posted_at"]).dt.total_seconds() / 86400
    df["age_days"] = age.fillna(0)

    df["log_reactions"] = np.log1p(df["reactions"].clip(lower=0))
    g = df.groupby("author")["log_reactions"]

    df["level"] = g.transform(prev_median)
    df["hist_mean"] = g.transform(prev_mean)
    df["hist_std"] = g.transform(prev_std)
    df["hist_min"] = g.transform(prev_min)
    df["hist_max"] = g.transform(prev_max)
    df["hist_iqr"] = g.transform(prev_q75) - g.transform(prev_q25)
    df["hist_n"] = df.groupby("author").cumcount()

    df["recent_median"] = g.transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).median())
    df["recent_vs_level"] = df["recent_median"] - df["level"]

    df["resid"] = df["log_reactions"] - df["level"]
    df["last_resid"] = df.groupby("author")["resid"].shift(1).fillna(0)
    df["resid_mean_3"] = df.groupby("author")["resid"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()).fillna(0)

    prev_time = df.groupby("author")["posted_at"].shift(1)
    df["days_since_prev"] = (
        (df["posted_at"] - prev_time).dt.total_seconds() / 86400).fillna(-1)

    for feat in ("n_words", "n_hashtags", "n_mentions", "n_urls", "n_emoji"):
        med = df.groupby("author")[feat].transform(prev_median).fillna(0)
        df["hist_median_" + feat] = med
        df["rel_" + feat] = np.log1p(df[feat]) - np.log1p(med.clip(lower=0))

    cts = df["content_type"].fillna("unknown").astype(str)
    order = {v: i for i, v in enumerate(cts.value_counts().index)}
    df["content_type_code"] = cts.map(order).astype(int)

    df = df.merge(media, on=["author", "post"], how="left")
    for col in ("n_images", "n_videos", "n_pdfs"):
        df[col] = df[col].fillna(0).astype(int)
    df["has_media"] = ((df["n_images"] + df["n_videos"] + df["n_pdfs"]) > 0).astype(int)

    df["n_paragraphs"] = text.str.count(r"\n\s*\n") + 1
    df["avg_word_len"] = (text.str.replace(r"\s+", "", regex=True).str.len()
                        / df["n_words"].clip(lower=1)).fillna(0)
    df["n_caps_words"] = text.str.count(r"\b[A-Z]{2,}\b")
    df["has_numbers"] = text.str.contains(r"\d", regex=True).astype(int)
    df["has_money"] = text.str.contains(r"[$€₽₸£¥]", regex=True).astype(int)
    df["recent_std"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).std())
    df["resid_mean_10"] = df.groupby("author")["resid"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()).fillna(0)

    dense_cols = [
        "level", "hist_mean", "hist_std", "hist_min", "hist_max", "hist_iqr",
        "hist_n", "recent_median", "recent_vs_level",
        "last_resid", "resid_mean_3", "days_since_prev",
        "log_followers", "age_days",
        "n_chars", "n_words", "n_lines", "n_hashtags", "n_mentions",
        "n_urls", "n_orgs", "n_people", "n_emoji",
        "has_question", "has_exclaim", "is_textless", "hour", "weekday",
        "hist_median_n_words", "hist_median_n_hashtags", "hist_median_n_mentions",
        "hist_median_n_urls", "hist_median_n_emoji",
        "rel_n_words", "rel_n_hashtags", "rel_n_mentions", "rel_n_urls", "rel_n_emoji",
        "n_images", "n_videos", "n_pdfs", "has_media", "content_type_code",
        "n_paragraphs", "avg_word_len", "n_caps_words", "has_numbers", "has_money",
        "recent_std", "resid_mean_10",
    ]
    return df, dense_cols


def build_text_matrix(train_texts, eval_texts, word_params, char_params, n_dims):
    word_vec = TfidfVectorizer(**word_params)
    char_vec = TfidfVectorizer(**char_params)

    X_tr = sp.hstack([word_vec.fit_transform(train_texts), char_vec.fit_transform(train_texts)]).tocsr()
    X_ev = sp.hstack([word_vec.transform(eval_texts), char_vec.transform(eval_texts)]).tocsr()

    n_features = X_tr.shape[1]
    svd = TruncatedSVD(n_components=n_dims, random_state=42, n_iter=10, algorithm="randomized", tol=1e-5)
    X_tr = svd.fit_transform(X_tr)
    X_ev = svd.transform(X_ev)
    print(f"SVD: from {n_features} to {n_dims} dims")
    return X_tr, X_ev


def load_image_features(data_dir, max_per_post=3):
    path = Path(data_dir) / "media" / "manifest.jsonl"
    empty = pd.DataFrame(columns=[
        "author", "post", "img_width", "img_height", "img_aspect",
        "img_size_kb", "img_brightness", "img_saturation", "img_colors",
    ])
    if not path.exists():
        print("no manifest, image features = 0")
        return empty

    cache = {}
    rows = []
    per_post = {}

    with open(path, encoding="utf-8") as f, tqdm(f, desc="image features") as pbar:
        for line in pbar:
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            if m.get("from_reshared_post"):
                continue
            p = m.get("path") or m.get("file") or m.get("filename")
            if not p or not str(p).lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue

            key = (m["author"], m["post"])
            if per_post.get(key, 0) >= max_per_post:
                continue
            per_post[key] = per_post.get(key, 0) + 1

            if p not in cache:
                full = Path(data_dir) / "media" / p
                try:
                    img = Image.open(full)
                    img.draft("RGB", (64, 64))
                    img = img.convert("RGB")
                    w, h = img.size
                    small = img.resize((64, 64))
                    pixels = list(small.getdata())
                    brightness = sum(sum(px) for px in pixels) / (3 * len(pixels))
                    saturation = sum(max(px) - min(px) for px in pixels) / len(pixels)
                    colors = len(small.getcolors(64 * 64) or [])
                    size_kb = full.stat().st_size / 1024 if full.exists() else 0
                    cache[p] = {
                        "img_width": w, "img_height": h,
                        "img_aspect": w / h if h else 1.0,
                        "img_size_kb": size_kb,
                        "img_brightness": brightness,
                        "img_saturation": saturation,
                        "img_colors": colors,
                    }
                except Exception:
                    cache[p] = None

            if cache[p]:
                rows.append({"author": m["author"], "post": m["post"], **cache[p]})

    if not rows:
        return empty

    df = pd.DataFrame(rows)
    return df.groupby(["author", "post"]).mean().reset_index()


def load_clip_features(data_dir, cache_path="clip_cache.npz"):
    from sentence_transformers import SentenceTransformer

    cache = Path(cache_path)
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        posts = list(data["posts"])
        print(f"loaded CLIP cache: {len(posts)} posts")
        df = pd.DataFrame(data["embs"],
                          columns=[f"clip_{i}" for i in range(data["embs"].shape[1])])
        df["author"] = [p[0] for p in posts]
        df["post"] = [p[1] for p in posts]
        return df

    path = Path(data_dir) / "media" / "manifest.jsonl"
    if not path.exists():
        return pd.DataFrame()
    post_to_files = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            if m.get("from_reshared_post"):
                continue
            p = m.get("path") or m.get("file") or m.get("filename")
            if not p or not str(p).lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            key = (m["author"], m["post"])
            post_to_files.setdefault(key, []).append(p)

    if not post_to_files:
        return pd.DataFrame()

    print(f"encoding {sum(len(v) for v in post_to_files.values())} images for {len(post_to_files)} posts ...")
    model = SentenceTransformer("clip-ViT-B-32")
    unique_files = list({p for files in post_to_files.values() for p in files})
    file_to_emb = {}
    batch_size = 32
    for i in range(0, len(unique_files), batch_size):
        batch_paths = unique_files[i:i + batch_size]
        imgs = []
        valid_paths = []
        for p in batch_paths:
            full = Path(data_dir) / "media" / p
            try:
                img = Image.open(full).convert("RGB")
                img.thumbnail((224, 224))
                imgs.append(img)
                valid_paths.append(p)
            except Exception:
                continue
        if not imgs:
            continue
        embs = model.encode(imgs, batch_size=16, show_progress_bar=False,
                            normalize_embeddings=True)
        for p, e in zip(valid_paths, embs):
            file_to_emb[p] = e
        if (i // batch_size) % 10 == 0:
            print(f"  {i + len(batch_paths)}/{len(unique_files)}")

    rows = []
    post_keys = []
    for (author, post), files in post_to_files.items():
        vecs = [file_to_emb[p] for p in files if p in file_to_emb]
        if vecs:
            rows.append(np.mean(vecs, axis=0))
            post_keys.append((author, post))

    if not rows:
        return pd.DataFrame()

    embs = np.vstack(rows)
    np.savez_compressed(cache, embs=embs, posts=post_keys)
    print(f"saved cache to {cache_path}")
    df = pd.DataFrame(embs, columns=[f"clip_{i}" for i in range(embs.shape[1])])
    df["author"] = [k[0] for k in post_keys]
    df["post"] = [k[1] for k in post_keys]
    return df

def build_bm25_features(df, k=5):
    from rank_bm25 import BM25Okapi

    df = df.copy().reset_index(drop=True)
    new = {name: [0.0] * len(df)
           for name in ("bm25_max", "bm25_mean", "bm25_resid", "bm25_logreact")}

    def tok(t):
        return [w for w in re.split(r"\W+", str(t).lower()) if len(w) >= 3]

    for _, idx in df.groupby("author").groups.items():
        idx = sorted(idx)
        tokens = [tok(df.at[i, "text"]) for i in idx]
        resid = [df.at[i, "resid"] for i in idx]
        logr = [df.at[i, "log_reactions"] for i in idx]

        for pos in range(5, len(idx)):
            past = [t for t in tokens[:pos] if t]
            q = tokens[pos]
            if len(past) < 3 or not q:
                continue

            scores = BM25Okapi(past).get_scores(q)
            top = np.argsort(scores)[-min(k, len(scores)):]
            row = idx[pos]

            new["bm25_max"][row] = float(scores.max())
            new["bm25_mean"][row] = float(scores.mean())
            top_resid = [resid[j] for j in top if not np.isnan(resid[j])]
            if top_resid:
                new["bm25_resid"][row] = float(np.mean(top_resid))
            new["bm25_logreact"][row] = float(np.mean([logr[j] for j in top]))

    for name, values in new.items():
        df[name] = values
    return df