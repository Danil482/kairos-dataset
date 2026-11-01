from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

CONTEXT_NUM = ["e5_profile_similarity", "e5_previous_similarity", "e5_recent5_similarity",
               "e5_history_similarity", "e5_history_posts"]


def cosine(left, right):
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(np.einsum("ij,ij->i", left, right), denominator,
                     out=np.full(len(left), np.nan), where=denominator > 0)


def causal_context(vectors):
    n = len(vectors)
    sums = np.vstack([np.zeros((1, vectors.shape[1])), np.cumsum(vectors, axis=0)])
    previous = np.vstack([np.zeros((1, vectors.shape[1])), vectors[:-1]])
    recent = sums[np.arange(n)] - sums[np.maximum(0, np.arange(n) - 5)]
    return np.column_stack([cosine(vectors, previous), cosine(vectors, recent),
                            cosine(vectors, sums[:-1]), np.arange(n)])


class EmbeddingStore:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.info = json.loads((self.folder / "encoder.json").read_text())
        keys = pd.read_csv(self.folder / "posts.csv")
        profiles = pd.read_csv(self.folder / "profiles.csv")
        self.post_index = pd.MultiIndex.from_frame(keys[["author", "post"]])
        self.profile_index = pd.Index(profiles.author)
        if self.post_index.has_duplicates or self.profile_index.has_duplicates:
            raise ValueError("Duplicate embedding keys")
        self.posts = np.load(self.folder / "posts.npy", mmap_mode="r")
        self.profiles = np.load(self.folder / "profiles.npy", mmap_mode="r")
        if len(self.posts) != len(keys) or len(self.profiles) != len(profiles):
            raise ValueError("Embedding vectors do not match their keys")
        self.history_keys = None
        self.history = None
        if self.info.get("history_input_sha256"):
            self.history_keys = pd.read_csv(self.folder / "history.csv")
            self.history_keys["posted_at"] = pd.to_datetime(self.history_keys.posted_at, utc=True, format="mixed")
            self.history = np.load(self.folder / "history.npy", mmap_mode="r")
            if len(self.history) != len(self.history_keys):
                raise ValueError("Historical embedding vectors do not match their keys")

    def arrays(self, table):
        post_rows = self.post_index.get_indexer(pd.MultiIndex.from_frame(table[["author", "post"]]))
        profile_rows = self.profile_index.get_indexer(table.author)
        if (post_rows < 0).any() or (profile_rows < 0).any():
            raise ValueError("Embedding coverage is incomplete; rerun embed_lab1.py on this release")
        return self.posts[post_rows], self.profiles[profile_rows]

    def context(self, table):
        posts, profiles = self.arrays(table)
        result = np.full((len(table), len(CONTEXT_NUM)), np.nan, dtype=np.float32)
        result[:, 0] = cosine(posts, profiles)
        timeline = table[["author", "posted_at"]].reset_index(drop=True)
        timeline["history_order"] = table.hist_n.to_numpy() if "hist_n" in table else np.arange(len(table))
        if self.history_keys is not None:
            mask = self.history_keys.author.isin(table.author).to_numpy()
            timeline = pd.concat([timeline, self.history_keys.loc[
                mask, ["author", "posted_at", "history_order"]]], ignore_index=True)
            posts = np.vstack([posts, self.history[mask]])
        context = np.full((len(timeline), 4), np.nan, dtype=np.float32)
        ordered = timeline.sort_values(["author", "posted_at", "history_order"], kind="mergesort")
        for _, positions in ordered.groupby("author", sort=False).groups.items():
            positions = np.asarray(positions, dtype=int)
            context[positions] = causal_context(posts[positions])
        result[:, 1:] = context[:len(table)]
        return pd.concat([table.reset_index(drop=True), pd.DataFrame(result, columns=CONTEXT_NUM)], axis=1)


class EmbeddingProjection:
    def __init__(self, post_components=64, profile_components=16):
        self.post_components = post_components
        self.profile_components = profile_components

    def fit(self, table, store):
        posts, _ = store.arrays(table)
        _, profiles = store.arrays(table.drop_duplicates("author"))
        self.post_pca = PCA(n_components=min(self.post_components, *posts.shape), random_state=42)
        self.profile_pca = PCA(n_components=min(self.profile_components, *profiles.shape), random_state=42)
        self.post_pca.fit(posts)
        self.profile_pca.fit(profiles)
        return self

    def transform(self, table, store):
        posts, profiles = store.arrays(table)
        values = np.hstack([self.post_pca.transform(posts), self.profile_pca.transform(profiles)])
        names = [f"e5_post_pc{i:03d}" for i in range(self.post_pca.n_components_)] + [
            f"e5_profile_pc{i:03d}" for i in range(self.profile_pca.n_components_)]
        return pd.concat([table.reset_index(drop=True), pd.DataFrame(values, columns=names)], axis=1), names
