"""Text baselines and strictly earlier, within-author vector histories.

TF-IDF/SVD must be fitted separately on each fold's labelled fit texts. The
resulting transformer can then encode unlabelled history and held-out authors;
those rows must never be passed to its fit method.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_array, check_is_fitted


SIMILARITY_SUFFIXES = (
    "cosine_past_mean", "cosine_last5_mean", "cosine_last", "cosine_max",
    "cosine_topk_mean", "valid_history_count", "missing",
)
HISTORICAL_OUTCOME_SUFFIXES = ("neighbor_log_reactions_mean_minus_level",)


def history_feature_columns(prefix: str = "semantic") -> tuple[list[str], list[str]]:
    """Separate pure similarity/availability from historical reaction features."""
    return ([f"{prefix}_{name}" for name in SIMILARITY_SUFFIXES],
            [f"{prefix}_{name}" for name in HISTORICAL_OUTCOME_SUFFIXES])


def _vectorizer(config: dict) -> TfidfVectorizer:
    settings = config.get("tfidf", config)
    analyzer = settings.get("analyzer", "word")
    ngrams = (1, 2) if analyzer == "word" else (3, 5)
    return TfidfVectorizer(
        analyzer=analyzer, ngram_range=tuple(settings.get("ngram_range", ngrams)),
        min_df=settings.get("min_df", 3), max_df=settings.get("max_df", 1.0),
        max_features=settings.get("max_features", 60000),
        sublinear_tf=settings.get("sublinear_tf", True),
        lowercase=settings.get("lowercase", True), dtype=np.float32,
    )


def make_text_model(config: dict) -> Pipeline:
    """Word/character TF-IDF followed by balanced logistic regression.

    Accepts flat settings, or dictionaries named ``tfidf`` and ``classifier``.
    Inputs are strings (for example a Series), not an entire feature table.
    """
    settings = config.get("classifier", config)
    return Pipeline([
        ("tfidf", _vectorizer(config)),
        ("clf", LogisticRegression(
            C=settings.get("C", 1.0), class_weight=settings.get("class_weight", "balanced"),
            max_iter=settings.get("max_iter", 1000), tol=settings.get("tol", 1e-4),
            random_state=config.get("seed", 42),
        )),
    ])


class AdaptiveSVD(TransformerMixin, BaseEstimator):
    """TruncatedSVD with a smaller rank only when a tiny fixture requires it.

    A one-word vocabulary uses its single TF-IDF coordinate unchanged because
    sklearn's TruncatedSVD requires at least two input features. Outputs retain
    their magnitude; cosine normalization belongs to history_vector_features.
    """

    def __init__(self, n_components: int = 128, n_iter: int = 7, random_state: int = 42):
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y=None):
        if not isinstance(self.n_components, int) or self.n_components < 1:
            raise ValueError("n_components must be a positive integer")
        X = check_array(X, accept_sparse=("csr", "csc"))
        self.n_features_in_ = X.shape[1]
        self.n_components_ = min(self.n_components, X.shape[0], X.shape[1])
        self.svd_ = None
        if X.shape[1] > 1:
            self.svd_ = TruncatedSVD(n_components=self.n_components_, n_iter=self.n_iter,
                                     random_state=self.random_state).fit(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        X = check_array(X, accept_sparse=("csr", "csc"))
        if X.shape[1] != self.n_features_in_:
            raise ValueError("SVD input vocabulary changed after fitting")
        if self.svd_ is None:
            result = X.toarray() if hasattr(X, "toarray") else X
        else:
            result = self.svd_.transform(X)
        return np.asarray(result, dtype=np.float32)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "n_components_")
        return np.asarray([f"svd_{i}" for i in range(self.n_components_)], dtype=object)


def fit_svd(texts, config: dict) -> Pipeline:
    """Fit TF-IDF -> unnormalized SVD on labelled fit texts only.

    Accepts flat settings or ``tfidf`` / ``svd`` dictionaries. The caller owns
    the author split: no other texts or labels are read by this function.
    """
    settings = config.get("svd", config)
    transformer = Pipeline([
        ("tfidf", _vectorizer(config)),
        ("svd", AdaptiveSVD(n_components=settings.get("n_components", settings.get("svd_components", 128)),
                             n_iter=settings.get("n_iter", settings.get("svd_n_iter", 7)),
                             random_state=config.get("seed", 42))),
    ])
    return transformer.fit(texts)


def history_vector_features(history: pd.DataFrame, vectors: np.ndarray, k: int = 3,
                            prefix: str = "semantic") -> pd.DataFrame:
    """Describe each post against strictly earlier own posts of its author.

    ``vectors`` must have the same row order as ``history``. Prefer content_text
    when present, otherwise use text. Empty text, the __empty__ sentinel,
    all-zero/OOV vectors and non-finite vectors never enter a text history.
    The last-five centroid uses the five most recent *valid* predecessors.
    Top-k ties prefer the most recent predecessor, then the post key.

    Missing similarities/outcomes are NaN. valid_history_count still describes
    the available past when the current text is missing. missing is 1 when the
    current vector is invalid or no valid predecessor exists. Historical
    reaction features read only the selected predecessors' log_reactions and
    the current level, never the current reaction or any label.
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    required = {"author", "post", "posted_at", "log_reactions", "level"}
    text_column = "content_text" if "content_text" in history else "text"
    required.add(text_column)
    if not required <= set(history):
        raise ValueError(f"History is missing columns: {sorted(required - set(history))}")
    if history[["author", "post"]].isna().any().any() or history.duplicated(["author", "post"]).any():
        raise ValueError("History must have unique, nonmissing (author, post) keys")

    frame = history.reset_index(drop=True)
    dates = pd.to_datetime(frame["posted_at"], utc=True, errors="raise")
    if dates.isna().any():
        raise ValueError("History timestamps must not be missing")
    times = dates.astype("int64").to_numpy()
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != len(frame) or values.shape[1] == 0:
        raise ValueError("Vectors must be a nonempty-width 2D array matching history rows")

    texts = frame[text_column].fillna("").astype(str).str.strip()
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    valid = ((texts.ne("") & texts.ne("__empty__")).to_numpy()
             & np.isfinite(values).all(axis=1) & (norms > 0))
    unit = np.zeros_like(values)
    unit[valid] = values[valid] / norms[valid, None]
    reactions = pd.to_numeric(frame["log_reactions"], errors="raise").to_numpy(dtype=float)
    levels = pd.to_numeric(frame["level"], errors="raise").to_numpy(dtype=float)
    similarity_cols, outcome_cols = history_feature_columns(prefix)
    columns = similarity_cols + outcome_cols
    result = np.full((len(frame), len(columns)), np.nan, dtype=float)
    result[:, 5] = 0  # valid_history_count
    result[:, 6] = 1  # missing

    def centroid_similarity(current, past):
        centroid = unit[past].mean(axis=0, dtype=np.float64)
        norm = np.linalg.norm(centroid)
        return float(np.clip(unit[current] @ centroid / norm, -1, 1)) if norm > 0 else np.nan

    for indices in frame.groupby("author", sort=False).indices.values():
        # Date ties have no causal order: exclude every equal timestamp below.
        order = np.lexsort((frame.iloc[indices]["post"].astype(str).to_numpy(), times[indices]))
        indices = indices[order]
        group_times, group_valid = times[indices], valid[indices]
        cosines = np.clip(unit[indices] @ unit[indices].T, -1, 1)
        for local, current in enumerate(indices):
            earlier = np.flatnonzero((group_times < group_times[local]) & group_valid)
            result[current, 5] = len(earlier)
            if not valid[current] or not len(earlier):
                continue
            result[current, 6] = 0
            past = indices[earlier]
            similarities = cosines[local, earlier]
            result[current, 0] = centroid_similarity(current, past)
            result[current, 1] = centroid_similarity(current, past[-5:])
            result[current, 2] = similarities[-1]
            result[current, 3] = similarities.max()
            # Reverse chronological order makes stable-sort ties prefer recency.
            ranked = np.argsort(-similarities[::-1], kind="stable")[:k]
            neighbors = past[::-1][ranked]
            result[current, 4] = similarities[::-1][ranked].mean()
            outcomes = reactions[neighbors]
            outcomes = outcomes[np.isfinite(outcomes)]
            if len(outcomes) and np.isfinite(levels[current]):
                result[current, 7] = outcomes.mean() - levels[current]

    output = frame[["author", "post"]].copy()
    output[columns] = result
    output[f"{prefix}_valid_history_count"] = output[f"{prefix}_valid_history_count"].astype("int32")
    output[f"{prefix}_missing"] = output[f"{prefix}_missing"].astype("int8")
    return output
