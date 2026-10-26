"""Export the measured semantic comparison and neural training curves."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "semantic"


def main():
    report = json.loads((OUT / "comparison.json").read_text())
    names = ["catboost_svm_fusion", "e5_logistic", "e5_mlp", "e5_history_mlp", "e5_history_catboost", "semantic_fusion"]
    labels = ["Прежний лучший вариант", "E5 + логистическая регрессия", "E5 + нейросеть", "E5 + история + нейросеть", "E5 + история + CatBoost", "Объединение прогнозов"]
    values = [report["oof"][name]["macro_f1"] for name in names]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.2, 1]})
    bars = axes[0].barh(labels, values, color=["#8895a7", "#5d9bc5", "#5d9bc5", "#3c9984", "#257462", "#7055aa"])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, .73)
    axes[0].set_xlabel("Macro-F1 на отложенных авторах (OOF)")
    axes[0].set_title("Сравнение на 71 395 постах", loc="left", weight="bold")
    for bar, value in zip(bars, values):
        axes[0].text(value+.009, bar.get_y()+bar.get_height()/2, f"{value:.4f}", va="center")
    for number, color in zip((1, 2, 3), ("#2e77ad", "#bb6538", "#30866d")):
        saved = json.loads((OUT / f"fold_{number}" / "e5_history_mlp_learning_curve.json").read_text())
        history = saved["history"]
        axes[1].plot([row["epoch"] for row in history], [row["inner_macro_f1"] for row in history], color=color, label=f"Разбиение {number}")
        best = history[saved["best_epoch"]-1]
        axes[1].scatter([best["epoch"]], [best["inner_macro_f1"]], color=color, s=55, marker="o", zorder=3)
    axes[1].set_title("Обучение нейросети: E5 + история", loc="left", weight="bold")
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("Macro-F1 на внутренних calibration-авторах")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=.2)
    figure.suptitle("Смысловые признаки текста в задаче Kairos", x=.02, ha="left", weight="bold", fontsize=15)
    figure.text(.02, .025, "Проверка внутри официального train. Точки справа — выбранные эпохи; официальный test на этом этапе не оценивался.", fontsize=9, color="#555555")
    figure.tight_layout(rect=[0, .065, 1, .94])
    figure.savefig(OUT / "semantic_results.png", dpi=170, bbox_inches="tight", facecolor="white")
    figure.savefig(OUT / "semantic_results.svg", bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print("Saved results/semantic/semantic_results.png and .svg")


if __name__ == "__main__":
    main()
