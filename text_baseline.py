"""Compare text models on held-out TRAIN authors; refit the winner for TEST.

    python text_baseline.py --prepared data/prepared
    python text_baseline.py --prepared data/prepared --embeddings data/embeddings
    python text_baseline.py --prepared data/prepared --embeddings data/embeddings --stage final

The final stage requires a matching validation report. It never reads test labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import baseline
from kairos import features
from kairos.metrics import score

SEED = 42
KEYS = ["author", "post"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_inputs(prepared: Path, embeddings: Path | None):
    frames = {}
    fingerprints = {}
    for split in ("train", "test"):
        path = prepared / f"{split}.csv.gz"
        fingerprints[split] = sha256(path)
        frame = pd.read_csv(path, keep_default_na=False)
        required = set(KEYS + ["text"] + baseline.NUMERIC + features.CONTENT_CAT)
        if split == "train":
            required.add("label")
        if not required <= set(frame.columns) or frame.empty:
            raise ValueError(f"{path}: missing required columns or no rows")
        if frame.duplicated(KEYS).any() or (frame[KEYS] == "").any().any():
            raise ValueError(f"{path}: duplicate or empty post keys")
        for name in baseline.NUMERIC:
            frame[name] = pd.to_numeric(frame[name], errors="raise")
        frames[split] = frame
    train, test = frames["train"], frames["test"]
    if "label" in test:
        raise ValueError("Prepared test must not contain labels; run prepare_text.py")
    if not train["label"].isin([0, 1, 2]).all():
        raise ValueError("Training labels must be 0, 1, 2")
    if set(train.author) & set(test.author):
        raise ValueError("An author appears in both official train and test")
    embedding_columns = []
    if embeddings is not None:
        meta = json.loads((embeddings / "metadata.json").read_text())
        if meta["input_sha256"] != fingerprints:
            raise ValueError("Embeddings belong to different prepared files")
        for split, frame in frames.items():
            path = embeddings / f"{split}.npy"
            if sha256(path) != meta["output_sha256"][split]:
                raise ValueError(f"{path}: embedding checksum mismatch")
            values = np.load(path, allow_pickle=False)
            if (values.ndim != 2 or values.shape != (len(frame), meta["dimensions"])
                    or meta["rows"][split] != len(frame) or not np.isfinite(values).all()):
                raise ValueError(f"{path}: invalid embedding shape or values")
            embedding_columns = [f"embedding_{i}" for i in range(values.shape[1])]
            frames[split] = pd.concat([
                frame, pd.DataFrame(values, columns=embedding_columns, index=frame.index)
            ], axis=1)
        fingerprints["embeddings"] = sha256(embeddings / "metadata.json")
    return frames["train"], frames["test"], embedding_columns, fingerprints


def make_model(name: str, embedding_columns: list[str]):
    if name == "baseline":
        return baseline.model()
    classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500,
                                    random_state=SEED)
    if name == "tfidf":
        return Pipeline([
            ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=3,
                                    max_features=60000, sublinear_tf=True,
                                    dtype=np.float32)),
            ("clf", classifier),
        ])
    if name == "embedding" and embedding_columns:
        numeric = baseline.NUMERIC + embedding_columns
        return Pipeline([
            ("prep", ColumnTransformer([
                ("num", Pipeline([("fill", SimpleImputer(strategy="median")),
                                   ("scale", StandardScaler())]), numeric),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                 features.CONTENT_CAT),
            ])),
            ("clf", classifier),
        ])
    raise ValueError(f"Unknown model: {name}")


def model_input(name: str, frame: pd.DataFrame):
    return frame["text"] if name == "tfidf" else frame


def probabilities(model, name: str, frame: pd.DataFrame) -> np.ndarray:
    if list(model.classes_) != [0, 1, 2]:
        raise ValueError("Each training partition must contain all three classes")
    return model.predict_proba(model_input(name, frame))


def candidates(with_embeddings: bool):
    options = {"baseline": {"baseline": 1.0}, "tfidf": {"tfidf": 1.0}}
    for weight in (0.25, 0.5, 0.75):
        options[f"baseline+tfidf:{weight}"] = {"baseline": 1 - weight, "tfidf": weight}
    if with_embeddings:
        options["embedding"] = {"embedding": 1.0}
        for weight in (0.25, 0.5, 0.75):
            options[f"baseline+embedding:{weight}"] = {
                "baseline": 1 - weight, "embedding": weight}
        options["baseline+tfidf+embedding"] = {"baseline": 0.5, "tfidf": 0.25,
                                               "embedding": 0.25}
    return options


def blend(weights: dict, predicted: dict) -> np.ndarray:
    return sum(weight * predicted[name] for name, weight in weights.items()).argmax(axis=1)


def configuration(fingerprints: dict, embedding_columns: list[str]) -> dict:
    return {
        "input_sha256": fingerprints, "script_sha256": sha256(Path(__file__)),
        "baseline_sha256": sha256(Path(baseline.__file__)), "seed": SEED,
        "validation_fraction_authors": 0.2, "embedding_dimensions": len(embedding_columns),
        "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "numpy": np.__version__, "pandas": pd.__version__,
    }


def validate(train, embedding_columns, config, output):
    fit_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                            random_state=SEED).split(train, groups=train.author))
    fit, val = train.iloc[fit_idx], train.iloc[val_idx]
    if set(fit.label) != {0, 1, 2} or set(val.label) != {0, 1, 2}:
        raise ValueError("All three classes must be present in fit and validation")
    print(f"fit: {len(fit):,} posts / {fit.author.nunique():,} authors; "
          f"validation: {len(val):,} posts / {val.author.nunique():,} authors", flush=True)
    predicted, timing = {}, {}
    names = ["baseline", "tfidf"] + (["embedding"] if embedding_columns else [])
    for name in names:
        started = time.perf_counter()
        print(f"Fitting {name}...", flush=True)
        model = make_model(name, embedding_columns)
        model.fit(model_input(name, fit), fit.label)
        predicted[name] = probabilities(model, name, val)
        timing[name] = round(time.perf_counter() - started, 2)
    results, raw_f1 = {}, {}
    options = candidates(bool(embedding_columns))
    for name, weights in options.items():
        labels = blend(weights, predicted)
        results[name] = score(val.label, labels)
        raw_f1[name] = float(f1_score(val.label, labels, average="macro", labels=[0, 1, 2]))
        print(f"{name:30s} macro-F1 {raw_f1[name]:.4f}", flush=True)
    winner = max(raw_f1, key=raw_f1.get)
    report = {"configuration": config, "selection_split": "held-out training authors",
              "n_fit": len(fit), "n_validation": len(val),
              "validation_authors": sorted(val.author.unique().tolist()),
              "results": results, "macro_f1_unrounded": raw_f1, "seconds": timing,
              "selected": winner, "weights": options[winner]}
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    val[KEYS].assign(label=blend(options[winner], predicted)).to_csv(
        output / "validation_predictions.csv", index=False)
    print(f"Selected: {winner}; saved {output / 'validation.json'}", flush=True)


def final(train, test, embedding_columns, config, output):
    report_path = output / "validation.json"
    if not report_path.exists():
        raise ValueError("Run the validation stage before final prediction")
    report = json.loads(report_path.read_text())
    if report["configuration"] != config:
        raise ValueError("Validation used different data, code or versions; rerun validation")
    options = candidates(bool(embedding_columns))
    weights = options[report["selected"]]
    if weights != report["weights"]:
        raise ValueError("Validation model weights do not match the fixed experiment grid")
    predicted = {}
    for name in weights:
        print(f"Refitting {name} on all {len(train):,} training posts...", flush=True)
        model = make_model(name, embedding_columns)
        model.fit(model_input(name, train), train.label)
        predicted[name] = probabilities(model, name, test)
    destination = output / "predictions.csv"
    test[KEYS].assign(label=blend(weights, predicted)).to_csv(destination, index=False)
    (output / "final.json").write_text(json.dumps({
        "configuration": config, "selected": report["selected"], "weights": weights,
        "n_train": len(train), "n_test": len(test),
        "predictions_sha256": sha256(destination), "test_labels_read": False,
    }, indent=2) + "\n")
    print(f"Saved {len(test):,} test predictions to {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/text"))
    parser.add_argument("--stage", choices=["validate", "final"], default="validate")
    args = parser.parse_args()
    train, test, columns, fingerprints = read_inputs(args.prepared, args.embeddings)
    config = configuration(fingerprints, columns)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "validate":
        validate(train, columns, config, args.output)
    else:
        final(train, test, columns, config, args.output)


if __name__ == "__main__":
    main()
