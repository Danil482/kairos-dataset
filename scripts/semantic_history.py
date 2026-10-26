"""Strictly earlier semantic author history, including unlabelled warm-up posts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from student_features import normalize_text
from scripts.analyze_train import extra_content, as_dict
from scripts.semantic_embeddings import (SOURCE, OUT as SEMANTIC, collect_texts, chunk_token_ids,
                                         fingerprint, save_json, CHUNK_TOKENS, MAX_CHUNKS)

OUT = ROOT / "results" / "semantic_history"
POLICY = dict(version=1, nearest_bank=100, neighbors=5, temperature=.05, half_life_days=180,
              shrinkage=5, chronology="same author, strictly earlier timestamp; ties excluded",
              windows=[5, 20], time_window_days=90)


def prepare_corpus():
    OUT.mkdir(exist_ok=True)
    payload = joblib.load(SOURCE)
    table = payload["table"].reset_index(drop=True)
    if set(table.split) != {"train"} or table.duplicated(["author", "post"]).any():
        raise ValueError("Expected unique official train posts")
    source_hash = fingerprint(SOURCE)
    if (OUT / "corpus_manifest.json").exists():
        saved = json.loads((OUT / "corpus_manifest.json").read_text())
        if saved["source_sha256"] != source_hash:
            raise ValueError("History corpus belongs to another training table")
        print("Prepared author-history corpus already available", flush=True)
        return
    authors, rows, digest = set(table.author), [], hashlib.sha256()
    for author in sorted(authors):
        raw = (ROOT / "data" / "posts" / f"{author}.json").read_bytes()
        digest.update(author.encode())
        digest.update(raw)
        feed = json.loads(raw)
        if feed["author"] != author:
            raise ValueError("Unexpected feed owner")
        for post in feed["posts"]:
            if as_dict(post.get("author")).get("code") != author:
                continue
            rows.append(dict(author=author, post=post["urn"], posted_at=post.get("postedDateTimestamp"),
                text=normalize_text(post.get("text") or ""), content_type=post.get("contentType") or "unknown",
                reactions=int(post.get("totalReactionCount") or 0)))
    corpus = pd.DataFrame(rows)
    corpus["posted_at"] = pd.to_datetime(corpus.posted_at, unit="ms", utc=True)
    if corpus.posted_at.isna().any() or corpus.duplicated(["author", "post"]).any():
        raise ValueError("Missing dates or duplicate history keys")
    corpus = corpus.merge(extra_content(authors)[["author", "post", "expanded_text"]],
                          on=["author", "post"], validate="one_to_one", how="left")
    check = table[["author", "post", "text", "expanded_text"]].merge(corpus, on=["author", "post"],
                how="left", suffixes=("_cached", ""), validate="one_to_one")
    if check.posted_at.isna().any() or not check.text.eq(check.text_cached).all() or not check.expanded_text.eq(check.expanded_text_cached).all():
        raise ValueError("Full history does not reproduce cached training texts")
    # Later unlabelled posts are unnecessary, even though the feature builder would exclude them.
    last_query = check.groupby("author").posted_at.max()
    corpus = corpus[corpus.posted_at.le(corpus.author.map(last_query))].reset_index(drop=True)
    joblib.dump(corpus, OUT / "corpus.joblib", compress=3)
    table[["author", "post"]].to_csv(OUT / "train_keys.csv", index=False)
    save_json(OUT / "corpus_manifest.json", dict(source_sha256=source_hash, feeds_sha256=digest.hexdigest(),
        corpus_sha256=fingerprint(OUT / "corpus.joblib"), rows=len(corpus), labelled_queries=len(table),
        additional_history_posts=len(corpus)-len(table), authors=len(authors), complete=True))
    print(f"History corpus: {len(corpus)} own posts, including {len(corpus)-len(table)} additional earlier posts", flush=True)


def encode_missing(texts, batch_size=32, cache_dir=OUT):
    """Same E5 pooling/chunk rules as the original cache, without external inference."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(4)
    spec = json.loads((SEMANTIC / "encoder.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(spec["local_path"], local_files_only=True, trust_remote_code=False)
    tokenizer.model_max_length = 10**9
    tokens = tokenizer(texts, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)["input_ids"]
    prefix = tokenizer("query: ", add_special_tokens=False)["input_ids"]
    records, owners, weights = [], [], []
    for owner, ids in enumerate(tokens):
        for chunk in chunk_token_ids(ids):
            prepared = tokenizer.build_inputs_with_special_tokens(prefix+chunk)
            if len(prepared) > 512:
                raise ValueError("E5 context overflow")
            records.append(dict(input_ids=prepared, attention_mask=[1]*len(prepared)))
            owners.append(owner)
            weights.append(max(len(chunk), 1))
    del tokens
    order = np.argsort([len(record["input_ids"]) for record in records], kind="stable")
    text_hash = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
    config = dict(text_sha256=text_hash, revision=spec["revision"], chunk_tokens=CHUNK_TOKENS, max_chunks=MAX_CHUNKS)
    cache_dir.mkdir(parents=True, exist_ok=True)
    progress_path = cache_dir / "embedding_progress.json"
    saved = json.loads(progress_path.read_text()) if progress_path.exists() else dict(config=config, completed_chunks=0)
    if saved["config"] != config:
        raise ValueError("Partial history embedding cache is incompatible")
    path = cache_dir / "new_chunks.npy"
    chunks = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype=np.float16, shape=(len(records),768))
    if chunks.shape != (len(records),768):
        raise ValueError("Partial chunk cache has the wrong shape")
    model = AutoModel.from_pretrained(spec["local_path"], local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=torch.float16, attn_implementation="sdpa").to("cuda").eval()
    start, last_print = saved["completed_chunks"], 0.
    print(f"Encoding {len(texts)} additional unique texts, {len(records)} chunks; starting at {start}", flush=True)
    for offset in range(start, len(records), batch_size):
        chosen = order[offset:offset+batch_size]
        batch = tokenizer.pad([records[i] for i in chosen], padding=True, pad_to_multiple_of=8, return_tensors="pt")
        batch = {key:value.to("cuda") for key,value in batch.items()}
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state.float()
            mask = batch["attention_mask"].unsqueeze(-1).bool()
            vectors = F.normalize(hidden.masked_fill(~mask,0.).sum(1)/mask.sum(1), p=2, dim=1).cpu().numpy()
        if not np.isfinite(vectors).all():
            raise ValueError("Invalid E5 embeddings")
        chunks[chosen] = vectors.astype(np.float16)
        stop = min(offset+batch_size, len(records))
        if stop % (batch_size*20) == 0 or stop == len(records):
            chunks.flush()
            save_json(progress_path, dict(config=config, completed_chunks=stop, total_chunks=len(records)))
        if time.monotonic()-last_print > 30 or stop == len(records):
            print(f"History E5 chunks {stop}/{len(records)}", flush=True)
            last_print = time.monotonic()
    values = np.zeros((len(texts),768), dtype=np.float32)
    for offset in range(0,len(records),2048):
        end = min(offset+2048,len(records))
        np.add.at(values, owners[offset:end], np.asarray(chunks[offset:end],dtype=np.float32)*np.asarray(weights[offset:end],dtype=np.float32)[:,None])
    values /= np.maximum(np.linalg.norm(values,axis=1,keepdims=True),1e-12)
    return values


def prepare_vectors():
    corpus_manifest = json.loads((OUT / "corpus_manifest.json").read_text())
    semantic_manifest = json.loads((SEMANTIC / "embedding_manifest.json").read_text())
    config = dict(corpus=corpus_manifest, semantic=semantic_manifest["config"], version=1)
    if (OUT / "embedding_manifest.json").exists():
        if json.loads((OUT / "embedding_manifest.json").read_text())["config"] != config:
            raise ValueError("History vectors do not match the corpus")
        print("Full-history vectors already available", flush=True)
        return
    corpus = joblib.load(OUT / "corpus.joblib")
    existing, _, _ = collect_texts(joblib.load(SOURCE)["table"])
    old = np.load(SEMANTIC / "unique_vectors.npy", mmap_mode="r")
    if len(old) != len(existing):
        raise ValueError("Existing E5 text/vector alignment changed")
    lookup = {text:i for i,text in enumerate(existing)}
    texts, mapping, clipped = collect_texts(corpus)
    values = np.zeros((len(texts),768), dtype=np.float32)
    missing = []
    for i,text in enumerate(texts):
        if text in lookup:
            values[i] = old[lookup[text]]
        else:
            missing.append(i)
    if missing:
        values[missing] = encode_missing([texts[i] for i in missing])
    np.save(OUT / "unique_vectors.npy", values)
    np.savez(OUT / "corpus_vector_indices.npz", **mapping)
    save_json(OUT / "embedding_manifest.json", dict(config=config, unique_texts=len(texts),
        reused_unique_texts=len(texts)-len(missing), new_unique_texts=len(missing), character_clipped_inputs=clipped, complete=True))
    print(f"Full-history E5 cache ready: {len(texts)-len(missing)} reused, {len(missing)} new texts", flush=True)


def cosine(a,b):
    scale = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip(np.dot(a,b)/scale,-1,1)) if scale > 1e-12 else np.nan


def history_features(posts, vectors, prefix, cutoffs):
    """Every row reads counters/vectors only at timestamps strictly below its own."""
    posts = posts.reset_index(drop=True)
    if len(posts) != len(vectors) or posts.posted_at.isna().any() or posts.duplicated(["author","post"]).any():
        raise ValueError("Invalid history alignment or timestamps")
    if not np.isfinite(vectors).all():
        raise ValueError("Non-finite history vectors")
    rows = [None]*len(posts)
    epoch = pd.Timestamp("1970-01-01",tz="UTC")
    for _, group in posts.sort_values(["author","posted_at","post"],kind="stable").groupby("author",sort=False):
        v = np.asarray(vectors[group.index],dtype=np.float64)
        present = np.linalg.norm(v,axis=1)>1e-8
        times = (group.posted_at-epoch).dt.total_seconds().to_numpy()
        logs = np.log1p(group.reactions.clip(lower=0).to_numpy(dtype=float))
        types = group.content_type.to_numpy()
        sums = np.vstack([np.zeros((1,v.shape[1])),np.cumsum(v,axis=0)])
        for i, position in enumerate(group.index):
            end = int(np.searchsorted(times,times[i],side="left"))
            indices = np.flatnonzero(present[:end])
            n = len(indices)
            shape = dict(current_available=int(present[i]), past_available=n,
                         past_missing=end-n, coherence=float(np.linalg.norm(sums[end])/n) if n else np.nan)
            # Define the full feature schema even for missing text or no history.
            shape.update({key:np.nan for key in ("centroid_all","centroid_5","centroid_20","centroid_90d",
                "last_similarity","max_100","mean_top5_100","mean_100","std_100","novelty_100",
                "near_repeat_count_100","novelty_relative_to_history","recent_vs_older","same_type_centroid",
                "nearest_gap_days","centroid_5_minus_all")})
            neighbor = {key:np.nan for key in ("top1_delta","top5_median_delta","weighted_delta",
                                               "weighted_std","below_probability","above_probability","weight_mass","effective_n")}
            if present[i] and n:
                current = v[i]
                shape["centroid_all"] = cosine(current,sums[end])
                for window in (5,20):
                    shape[f"centroid_{window}"] = cosine(current,v[indices[-window:]].sum(axis=0))
                start90 = int(np.searchsorted(times[:end], times[i]-90*86400,side="left"))
                shape["centroid_90d"] = cosine(current,sums[end]-sums[start90])
                shape["last_similarity"] = cosine(current,v[indices[-1]])
                bank = indices[-POLICY["nearest_bank"]:]
                similarity = np.clip(v[bank] @ current,-1,1)
                chosen = np.argsort(-similarity,kind="stable")[:POLICY["neighbors"]]
                selected = bank[chosen]
                shape.update(max_100=float(similarity.max()), mean_top5_100=float(similarity[chosen].mean()),
                    mean_100=float(similarity.mean()), std_100=float(similarity.std()), novelty_100=float(1-similarity.max()),
                    near_repeat_count_100=int(np.count_nonzero(similarity>.995)),
                    nearest_gap_days=float((times[i]-times[selected[0]])/86400),
                    centroid_5_minus_all=shape["centroid_5"]-shape["centroid_all"])
                if n>1:
                    pair_mean = (np.dot(sums[end],sums[end])-n)/(n*(n-1))
                    shape["novelty_relative_to_history"] = float(np.dot(current,sums[end])/n-pair_mean)
                if n>5:
                    shape["recent_vs_older"] = cosine(v[indices[-5:]].sum(axis=0),v[indices[:-5]].sum(axis=0))
                same = indices[types[indices]==types[i]]
                shape["same_type_centroid"] = cosine(current,v[same].sum(axis=0)) if len(same) else np.nan
                # Snapshot reactions of earlier posts only, no current outcomes/labels.
                level = float(np.median(logs[:end]))
                weight = np.exp((similarity[chosen]-similarity[chosen].max())/POLICY["temperature"])
                weight *= np.exp2(np.maximum(-(times[i]-times[selected])/(86400*POLICY["half_life_days"]),-1000))
                mass = float(weight.sum())
                if mass > 1e-100:
                    values = logs[selected]
                    neighbor.update(top1_delta=float(values[0]-level), top5_median_delta=float(np.median(values)-level),
                        weighted_delta=float(np.dot(weight,values-level)/(mass+POLICY["shrinkage"])),
                        weighted_std=float(np.sqrt(np.average((values-np.average(values,weights=weight))**2,weights=weight))),
                        below_probability=float((np.dot(weight,values<level+cutoffs[0])+1)/(mass+5)),
                        above_probability=float((np.dot(weight,values>level+cutoffs[1])+1)/(mass+5)),
                        weight_mass=mass, effective_n=float(mass*mass/np.dot(weight,weight)))
            rows[position] = {**{f"nov_{prefix}_{key}":value for key,value in shape.items()},
                              **{f"nbr_{prefix}_{key}":value for key,value in neighbor.items()}}
    return pd.DataFrame(rows)


def build_features():
    manifest = json.loads((OUT / "embedding_manifest.json").read_text())
    config = dict(embeddings=manifest["config"], policy=POLICY)
    if (OUT / "features_manifest.json").exists():
        if json.loads((OUT / "features_manifest.json").read_text())["config"] != config:
            raise ValueError("Semantic history feature cache differs")
        print("Semantic history features already complete", flush=True)
        return
    payload = joblib.load(SOURCE)
    table = payload["table"].reset_index(drop=True)
    corpus = joblib.load(OUT / "corpus.joblib")
    vectors = np.vstack([np.zeros((1,768),dtype=np.float32),np.load(OUT / "unique_vectors.npy")])
    parts = []
    with np.load(OUT / "corpus_vector_indices.npz") as mapping, threadpool_limits(limits=4):
        for field,prefix in (("text","own"),("expanded_text","expanded")):
            print(f"Building strictly earlier {prefix} semantic comparisons", flush=True)
            parts.append(history_features(corpus,vectors[mapping[field]+1],prefix,payload["cutoffs"]))
    features = pd.concat([corpus[["author","post"]],*parts],axis=1)
    features = table[["author","post"]].merge(features,on=["author","post"],how="left",validate="one_to_one")
    columns = [c for c in features if c not in ("author","post")]
    if features.nov_own_current_available.isna().any() or np.isinf(features[columns].to_numpy()).any():
        raise ValueError("Missing queries or infinite features")
    joblib.dump(features,OUT / "features.joblib",compress=3)
    save_json(OUT / "features_manifest.json",dict(config=config, rows=len(features),
        novelty_columns=[c for c in columns if c.startswith("nov_")],
        neighbor_columns=[c for c in columns if c.startswith("nbr_")],
        own_queries_with_history=int((features.nov_own_current_available.eq(1)&features.nov_own_past_available.gt(0)).sum()),
        expanded_queries_with_history=int((features.nov_expanded_current_available.eq(1)&features.nov_expanded_past_available.gt(0)).sum()),
        features_sha256=fingerprint(OUT / "features.joblib"),complete=True))
    print(f"Saved {len(columns)} history features for all {len(features)} labelled train posts",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only",action="store_true")
    args = parser.parse_args()
    prepare_corpus()
    if not args.prepare_only:
        prepare_vectors()
        build_features()


if __name__ == "__main__":
    main()
