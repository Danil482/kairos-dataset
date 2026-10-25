"""Pretty-print a comparison of all runs we've produced so far."""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _load(name: str) -> dict | None:
    p = RESULTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    rows = []

    base = _load("baseline.json")
    if base:
        for name, m in base["results"].items():
            rows.append((f"baseline/{name}", m))

    t1 = _load("tier1_cv.json")
    if t1:
        for name, info in t1["metrics"].items():
            rows.append((f"tier1/{name} (OOF)", info["oof"]))
            rows.append((f"tier1/{name} (test)", info["test"]))

    t2 = _load("tier2_cv.json")
    if t2:
        for name, info in t2["metrics"].items():
            rows.append((f"tier2/{name} (OOF)", info["oof"]))
            rows.append((f"tier2/{name} (test)", info["test"]))

    print(f"\n{'model':40s} {'macro-F1':>9s} {'QWK':>7s} {'accuracy':>9s}")
    print("-" * 70)
    for name, m in rows:
        print(f"{name:40s} {m['macro_f1']:9.4f} {m['qwk']:7.4f} {m['accuracy']:9.4f}")


if __name__ == "__main__":
    main()
