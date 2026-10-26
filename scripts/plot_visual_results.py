"""Export measured visual ablations and the empty-text diagnostic slice."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "visual"


def main():
    report = json.loads((OUT / "comparison.json").read_text())
    names = ["semantic_fusion", "e5_history_catboost", "media_metadata_catboost", "multimodal_catboost", "multimodal_mlp", "visual_fusion"]
    labels = ["Прежнее объединение с E5", "E5 + история + CatBoost", "+ количество и размеры картинок", "+ содержание картинок (SigLIP 2)", "Текст + картинки + нейросеть", "Новое объединение прогнозов"]
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios":[1.25,1]})
    colors = ["#8895a7", "#75998d", "#d3974e", "#267c69", "#5b8eb4", "#7858a6"]
    values = [report["oof"][name]["macro_f1"] for name in names]
    bars = axes[0].barh(labels, values, color=colors)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(values)+.08)
    axes[0].set_xlabel("Macro-F1, все 71 395 постов")
    axes[0].set_title("Контроль вклада содержания изображений", loc="left", weight="bold")
    for bar, value in zip(bars, values):
        axes[0].text(value+.008, bar.get_y()+bar.get_height()/2, f"{value:.4f}", va="center")
    selected = ["e5_history_catboost", "media_metadata_catboost", "multimodal_catboost", "visual_fusion"]
    for name, label, color in zip(selected, ["E5 + история", "+ метаданные", "+ SigLIP 2", "Объединение"], [colors[1],colors[2],colors[3],colors[5]]):
        scores = [report["slices"][group]["models"][name]["macro_f1"] for group in ("has_images", "empty_original_with_images")]
        axes[1].plot([0,1], scores, "o-", color=color, label=label)
    axes[1].set_xticks([0,1], ["Все посты с картинками", "Пустой собственный текст\n+ картинки"])
    axes[1].set_xlim(-.15,1.15)
    axes[1].set_ylabel("Macro-F1 внутри группы")
    axes[1].set_title("Результаты на группах постов", loc="left", weight="bold")
    axes[1].grid(axis="y", alpha=.2)
    axes[1].legend(frameon=False, fontsize=9)
    figure.suptitle("Kairos: текст, история автора и содержание изображений", x=.02, ha="left", weight="bold", fontsize=14)
    figure.text(.02,.02,"Три прежних разбиения по авторам внутри train. Официальный test не оценивался. Значения на группах не сравниваются как равные задачи.", fontsize=8.5, color="#555555")
    figure.tight_layout(rect=[0,.065,1,.94])
    for extension in ("png", "svg"):
        figure.savefig(OUT / f"visual_results.{extension}", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print("Saved results/visual/visual_results.png and .svg")


if __name__ == "__main__":
    main()
