"""Aggregate research figure; contains no raw post content or author identifiers."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scripts.priority_features import OUT

LABELS = dict(v2="Кандидат v2", history_type="История того же формата", history_quote="Тот же режим цитирования",
    recency_history="Вес недавней истории", additional_fields="Контроль: ещё 32 PCA текста",
    separate_fields="Цитата и документ отдельно", visual_history="Визуальная история",
    balanced_loss="Веса классов при обучении", ordinal="Два порядковых порога",
    blend_text="Текстовая ветка объединения", blend_visual="Визуальная ветка объединения",
    constant_blend="Постоянные веса объединения", adaptive_blend="Веса по доступности контента")


def main():
    report = json.loads((OUT / "comparison.json").read_text())
    names = list(report["oof"])
    scores = np.array([report["oof"][name]["macro_f1"] for name in names])
    empty = [report["slices"]["empty_original"]["macro_f1"][name] for name in names]
    reference = scores[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw=dict(width_ratios=[1.25, 1]))
    y = np.arange(len(names))
    colors = ["#163a5f" if name=="v2" else "#2a876e" if value>reference else "#8c9eaf" for name,value in zip(names,scores)]
    axes[0].scatter(scores, y, color=colors, s=55)
    axes[0].set_yticks(y, [LABELS[name] for name in names])
    axes[0].invert_yaxis()
    axes[0].set_xlim(min(scores)-.008, max(scores)+.006)
    axes[0].axvline(reference, color="#163a5f", linestyle="--", linewidth=1)
    for i, value in enumerate(scores):
        axes[0].text(value+.0004, i, f"{value:.4f}", va="center", fontsize=9, bbox=dict(facecolor="white", edgecolor="none", pad=.3))
    axes[0].set_title("Все посты: macro-F1")
    axes[1].scatter(empty, y, color=colors, s=55)
    axes[1].set_yticks(y, [""]*len(names))
    axes[1].invert_yaxis()
    axes[1].set_xlim(min(empty)-.018, max(empty)+.025)
    axes[1].axvline(empty[0], color="#163a5f", linestyle="--", linewidth=1)
    for i, value in enumerate(empty):
        axes[1].text(value+.001, i, f"{value:.4f}", va="center", fontsize=9, bbox=dict(facecolor="white", edgecolor="none", pad=.3))
    axes[1].set_title("Без собственного текста: macro-F1")
    for ax in axes:
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Развитие полного контента и истории автора", fontsize=17)
    fig.text(.5, .02, "71 395 постов · 3 внешних разбиения по авторам внутри train · официальный test не использован", ha="center", fontsize=10)
    fig.tight_layout(rect=[0,.04,1,.95])
    for extension in ("png", "svg"):
        fig.savefig(OUT / f"priority_results.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
