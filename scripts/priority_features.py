"""Frozen v2 snapshot and controlled train-only content/history extensions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from student_features import normalize_text
from scripts.analyze_train import as_dict, text_value
from scripts.semantic_embeddings import SOURCE, collect_texts, fingerprint, save_json
from scripts.semantic_history import OUT as HISTORY, encode_missing, cosine, history_features

OUT = ROOT / "results" / "priority_research"
POLICY = dict(version=1, shrinkage=5., neighbors=5, temperature=.05,
              recency_days=30., context_days=180., quote_pca=16, additional_pca=32,
              chronology="same author, strictly earlier; simultaneous posts excluded",
              visual_history="cached official train posts only; warm-up images not encoded",
              fields=["quote_text", "document_text", "auxiliary_text", "additional_text"],
              feature_experiments=["history_type", "history_quote", "recency_history",
                                   "additional_fields", "separate_fields", "visual_history"])


def freeze_v2():
    dest = ROOT / "results" / "candidates" / "v2"
    dest.mkdir(parents=True, exist_ok=True)
    files = [HISTORY / "comparison.json", HISTORY / "features_manifest.json", HISTORY / "oof_predictions.csv"]
    for number in (1, 2, 3):
        directory = HISTORY / f"fold_{number}"
        files += [directory / name for name in ("visual_neighbors.cbm", "visual_neighbors_schema.json",
                    "visual_neighbors_probabilities.npz", "semantic_pca.joblib", "visual_pca.joblib",
                    "report.json", "predictions.csv", "calibration_keys.csv")]
    # Preserve the implementation and pin the larger reusable feature/encoder caches by hash.
    files += [ROOT / "scripts" / name for name in ("history_models.py", "semantic_history.py",
              "semantic_models.py", "visual_models.py", "visual_embeddings.py", "semantic_embeddings.py")]
    files += [ROOT / "results" / name / "encoder.json" for name in ("semantic", "visual")]
    entries = []
    for source in files:
        relative = source.relative_to(ROOT)
        target = dest / relative
        digest = fingerprint(source)
        if target.exists() and fingerprint(target) != digest:
            raise ValueError("Frozen v2 differs from current source; do not overwrite the candidate")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        entries.append(dict(path=relative.as_posix(), sha256=digest))
    value = dict(candidate="v2", model="visual_neighbors", scope="Frozen three-fold development candidate, not a final all-train model",
                 official_test_evaluated=False, source_sha256=fingerprint(SOURCE), files=entries,
                 scores=json.loads((HISTORY / "comparison.json").read_text())["oof"]["visual_neighbors"])
    manifest = dest / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != value:
        raise ValueError("Frozen v2 manifest differs")
    save_json(manifest, value)
    print("v2 candidate snapshot verified", flush=True)


def join_fields(values):
    return normalize_text("\n".join(text_value(value) for value in values if text_value(value)))


def content_fields(post):
    quote = as_dict(post.get("resharedPost"))
    article, doc = as_dict(post.get("article")), as_dict(post.get("document"))
    qa, qd = as_dict(quote.get("article")), as_dict(quote.get("document"))
    entity, celebration = as_dict(post.get("entity")), as_dict(post.get("celebration"))
    extra = [qa.get("subtitle"), qd.get("title"), qd.get("transcript")]
    own = text_value(post.get("text"))
    return dict(quote_mode=0 if not quote else 1 if len(own) else 2,
        quote_text=join_fields([quote.get("text"), qa.get("title"), *extra]),
        document_text=join_fields([doc.get("title"), doc.get("transcript")]),
        auxiliary_text=join_fields([article.get("title"), article.get("subtitle"), entity.get("title"),
                                   entity.get("subtitle"), celebration.get("text")]),
        new_text=join_fields(extra))


def contextual_history(posts, vectors, mode, prefix, cutoffs):
    """Matched history with conservative shrinkage to the author's full past level."""
    posts = posts.reset_index(drop=True)
    if len(posts) != len(vectors) or posts.posted_at.isna().any() or posts.duplicated(["author", "post"]).any():
        raise ValueError("Invalid context history alignment")
    if mode not in ("type", "quote", "recency") or not np.isfinite(vectors).all():
        raise ValueError("Invalid history mode/vectors")
    rows = [None] * len(posts)
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    for _, group in posts.sort_values(["author", "posted_at", "post"], kind="stable").groupby("author", sort=False):
        vec = np.asarray(vectors[group.index], dtype=np.float64)
        present = np.linalg.norm(vec, axis=1) > 1e-8
        times = (group.posted_at-epoch).dt.total_seconds().to_numpy()
        logs = np.log1p(group.reactions.clip(lower=0).to_numpy(dtype=float))
        kinds, modes = group.content_type.to_numpy(), group.quote_mode.to_numpy()
        for i, position in enumerate(group.index):
            end = int(np.searchsorted(times, times[i], side="left"))
            bank = np.arange(end)
            if mode == "type":
                bank = bank[kinds[:end] == kinds[i]]
            elif mode == "quote":
                bank = bank[modes[:end] == modes[i]]
            row = dict(n=len(bank), mass=0., effective_n=0., mean_delta=0., below=0., above=0.,
                       vector_n=0, centroid=np.nan, max_similarity=np.nan, nearest_delta=np.nan,
                       neighbor_delta=0., neighbor_mass=0., weighted_std=np.nan)
            if end:
                level = float(np.median(logs[:end]))
                half_life = POLICY["recency_days"] if mode == "recency" else POLICY["context_days"]
                weight = np.exp2(np.maximum(-(times[i]-times[bank])/(86400*half_life), -1000))
                mass = float(weight.sum())
                shrink = POLICY["shrinkage"]
                row.update(mass=mass, effective_n=float(mass*mass/max(float(weight @ weight), 1e-100)),
                    mean_delta=float(weight @ (logs[bank]-level)/(mass+shrink)),
                    below=float((weight @ (logs[bank] < level+cutoffs[0]) + shrink*np.mean(logs[:end] < level+cutoffs[0]))/(mass+shrink)),
                    above=float((weight @ (logs[bank] > level+cutoffs[1]) + shrink*np.mean(logs[:end] > level+cutoffs[1]))/(mass+shrink)))
                valid = present[bank]
                selected = bank[valid]
                row["vector_n"] = len(selected)
                if present[i] and len(selected):
                    similarity = np.clip(vec[selected] @ vec[i], -1, 1)
                    best = np.argsort(-similarity, kind="stable")[:POLICY["neighbors"]]
                    near = selected[best]
                    weights = np.exp((similarity[best]-similarity[best].max())/POLICY["temperature"]) * weight[valid][best]
                    total = float(weights.sum())
                    row.update(centroid=cosine(vec[i], (vec[selected]*weight[valid, None]).sum(axis=0)),
                        max_similarity=float(similarity.max()), nearest_delta=float(logs[near[0]]-level),
                        neighbor_delta=float(weights @ (logs[near]-level)/(total+shrink)), neighbor_mass=total)
                    if total > 1e-100:
                        center = float(weights @ logs[near] / total)
                        row["weighted_std"] = float(np.sqrt(weights @ ((logs[near]-center)**2)/total))
            rows[position] = {prefix+"_"+key: value for key, value in row.items()}
    return pd.DataFrame(rows)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    freeze_v2()
    config = dict(policy=POLICY, source_sha256=fingerprint(SOURCE), corpus_sha256=fingerprint(HISTORY / "corpus.joblib"))
    protocol = OUT / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != config:
        raise ValueError("Research protocol/cache changed")
    save_json(protocol, config)
    complete = OUT / "features_manifest.json"
    if complete.exists():
        saved = json.loads(complete.read_text())
        if saved["config"] != config or saved["features_sha256"] != fingerprint(OUT / "features.joblib"):
            raise ValueError("Research feature cache changed")
        print("Priority features already complete", flush=True)
        return
    payload = joblib.load(SOURCE)
    table = payload["table"].reset_index(drop=True)
    corpus = joblib.load(HISTORY / "corpus.joblib")
    fields_path = OUT / "fields.joblib"
    if fields_path.exists():
        fields = joblib.load(fields_path)
    else:
        rows = []
        for author in sorted(set(corpus.author)):
            feed = json.loads((ROOT / "data" / "posts" / f"{author}.json").read_text(encoding="utf-8"))
            for post in feed["posts"]:
                if as_dict(post.get("author")).get("code") == author:
                    rows.append(dict(author=author, post=post["urn"], **content_fields(post)))
        fields = corpus[["author", "post", "expanded_text"]].merge(pd.DataFrame(rows), on=["author", "post"], validate="one_to_one")
        fields["additional_text"] = [join_fields([old, new]) for old, new in zip(fields.expanded_text, fields.new_text)]
        fields = fields.drop(columns=["expanded_text", "new_text"])
        joblib.dump(fields, fields_path, compress=3)
    if not fields[["author", "post"]].equals(corpus[["author", "post"]]):
        raise ValueError("Content fields are misaligned")
    corpus = pd.concat([corpus, fields.drop(columns=["author", "post"])], axis=1)
    indices = corpus.reset_index()[["author", "post", "index"]]
    query_positions = table[["author", "post"]].merge(indices, on=["author", "post"], validate="one_to_one")["index"].to_numpy()
    np.save(OUT / "query_positions.npy", query_positions)
    vectors = np.vstack([np.zeros((1,768), dtype=np.float32), np.load(HISTORY / "unique_vectors.npy")])
    mapping = np.load(HISTORY / "corpus_vector_indices.npz")
    expanded = vectors[mapping["expanded_text"]+1]
    chunks, groups = [], {}
    for mode, name in (("type", "history_type"), ("quote", "history_quote"), ("recency", "recency_history")):
        print(f"Building {name}", flush=True)
        with threadpool_limits(limits=4):
            block = contextual_history(corpus, expanded, mode, name, payload["cutoffs"]).iloc[query_positions].reset_index(drop=True)
        if mode == "quote":
            block["history_quote_mode"] = fields.quote_mode.iloc[query_positions].to_numpy()
        groups[name] = block.columns.tolist()
        chunks.append(block)
    print("Building visual history from existing content-image vectors", flush=True)
    image_vectors = np.zeros((len(corpus), 768), dtype=np.float32)
    image_vectors[query_positions] = np.load(ROOT / "results" / "visual" / "post_mean.npy")
    with threadpool_limits(limits=4):
        visual = history_features(corpus, image_vectors, "image", payload["cutoffs"]).iloc[query_positions].reset_index(drop=True)
    groups["visual_history"] = visual.columns.tolist()
    chunks.append(visual)
    # Encode source fields separately. Reuse exact previously encoded strings where possible.
    vector_path = OUT / "field_vectors.npz"
    if not vector_path.exists():
        old_texts, _, _ = collect_texts(corpus)
        old_lookup = {text: i+1 for i, text in enumerate(old_texts)}
        if len(old_texts)+1 != len(vectors):
            raise ValueError("Existing text/vector cache differs")
        query_fields = fields.iloc[query_positions]
        unique, lookup, field_indices = [], {}, {}
        for field in POLICY["fields"]:
            values = []
            for value in query_fields[field]:
                value = value[:20000].strip()
                if value and value not in lookup:
                    lookup[value] = len(unique)
                    unique.append(value)
                values.append(lookup[value] if value else -1)
            field_indices[field] = np.asarray(values)
        encoded = np.zeros((len(unique)+1, 768), dtype=np.float32)
        missing = []
        for i, value in enumerate(unique):
            if value in old_lookup:
                encoded[i+1] = vectors[old_lookup[value]]
            else:
                missing.append(i)
        print(f"Separate fields: {len(unique)} unique texts, {len(missing)} new E5 inputs", flush=True)
        if missing:
            encoded[np.asarray(missing)+1] = encode_missing([unique[i] for i in missing], cache_dir=OUT / "e5")
        np.savez_compressed(vector_path, **{field: encoded[idx+1] for field, idx in field_indices.items()})
        save_json(OUT / "field_embedding_manifest.json", dict(unique_texts=len(unique), new_texts=len(missing),
            sha256=fingerprint(vector_path), fields_sha256=fingerprint(fields_path)))
    field_vectors = np.load(vector_path)
    block = pd.DataFrame(index=range(len(table)))
    own = vectors[mapping["text"]+1][query_positions]
    for field in POLICY["fields"][:3]:
        valid = np.linalg.norm(field_vectors[field], axis=1) > 1e-8
        block[field+"_available"] = valid.astype(float)
        block[field+"_log_chars"] = np.log1p(fields[field].iloc[query_positions].str.len().to_numpy())
        sim = np.sum(own*field_vectors[field], axis=1)
        sim[~valid | (np.linalg.norm(own, axis=1) < 1e-8)] = np.nan
        block[field+"_own_similarity"] = sim
    groups["separate_fields"] = block.columns.tolist()
    chunks.append(block)
    extra = pd.DataFrame(dict(additional_changed=(fields.additional_text.iloc[query_positions].to_numpy() != table.expanded_text.to_numpy()).astype(float)))
    groups["additional_fields"] = extra.columns.tolist()
    chunks.append(extra)
    result = pd.concat([table[["author", "post"]], *chunks], axis=1)
    if result.columns.duplicated().any() or np.isinf(result.drop(columns=["author", "post"]).to_numpy()).any():
        raise ValueError("Invalid research features")
    joblib.dump(result, OUT / "features.joblib", compress=3)
    save_json(complete, dict(config=config, complete=True, groups=groups, rows=len(result),
        features_sha256=fingerprint(OUT / "features.joblib"), fields_sha256=fingerprint(fields_path),
        field_vectors_sha256=fingerprint(vector_path),
        coverage={field: int((np.linalg.norm(field_vectors[field], axis=1)>1e-8).sum()) for field in POLICY["fields"]},
        additional_changed=int(extra.additional_changed.sum()), visual_queries_with_history=int((visual.nov_image_past_available>0).sum())))
    print("All six content feature blocks ready", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        prepare()
