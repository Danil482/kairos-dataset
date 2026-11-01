from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from train_lab1 import blend, checked_table, tabular
from lab1_embeddings import CONTEXT_NUM, EmbeddingStore


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--model", type=Path, default=Path("results/lab1/model.joblib"))
    ap.add_argument("--output", type=Path, default=Path("results/lab1/reproduced.csv"))
    ap.add_argument("--embeddings", type=Path, default=Path(".cache/embeddings"))
    args = ap.parse_args()
    saved = joblib.load(args.model)
    table, numeric = checked_table(args.data)
    if saved.get("embedding_projection") is not None:
        store = EmbeddingStore(args.embeddings)
        if store.info["input_sha256"] != saved["embedding_info"]["input_sha256"]:
            raise SystemExit("Embedding input fingerprint differs from the trained model")
        if store.info.get("history_input_sha256") != saved["embedding_info"].get("history_input_sha256"):
            raise SystemExit("Historical embedding fingerprint differs from the trained model")
        table = store.context(table)
        table, extra = saved["embedding_projection"].transform(table, store)
        numeric = numeric + CONTEXT_NUM + extra
    if numeric != saved["numeric"]:
        raise SystemExit("Feature schema does not match trained model")
    test = table[table.split == "test"]
    tree_prob = saved["tree"].predict_proba(tabular(test, numeric))
    text_prob = saved["text"].predict_proba(test)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    test[["author", "post"]].assign(label=blend(tree_prob, text_prob, saved["config"])).to_csv(
        args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
