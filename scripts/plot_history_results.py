"""Export controlled semantic novelty and visual-combination results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "semantic_history"


def main():
    report = json.loads((OUT / "comparison.json").read_text())
    definitions = [("e5_history_catboost","E5 + обычная история"), ("text_novelty","+ смысловая новизна"),
        ("text_neighbors","+ реакции похожих прошлых постов"), ("multimodal_catboost","Прежняя модель с изображениями"),
        ("visual_novelty","Изображения + смысловая новизна"), ("visual_neighbors","Изображения + новизна + похожие посты")]
    definitions = [(name,label) for name,label in definitions if name in report["oof"]]
    values = [report["oof"][name]["macro_f1"] for name,_ in definitions]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    figure,axes = plt.subplots(1,2,figsize=(14,5.4),gridspec_kw={"width_ratios":[1.35,1]})
    colors = ["#9aa6b5","#4b91b6","#287a91","#9baf9d","#4b977b","#276e54"][:len(definitions)]
    bars = axes[0].barh([label for _,label in definitions],values,color=colors)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0,max(values)+.08)
    axes[0].set_xlabel("OOF macro-F1, все 71 395 постов")
    axes[0].set_title("Что добавляет сравнение с прошлым",loc="left",weight="bold")
    for bar,value in zip(bars,values):
        axes[0].text(value+.008,bar.get_y()+bar.get_height()/2,f"{value:.4f}",va="center")
    for name,label,color in [("text_novelty","Новизна текста","#4b91b6"),("text_neighbors","+ похожие посты","#287a91"),
                              ("visual_novelty","Новизна + картинки","#4b977b"),("visual_neighbors","Все признаки","#276e54")]:
        if name not in report["models_completed"]:
            continue
        axes[1].plot([1,2,3],[report["folds"][str(n)][name] for n in (1,2,3)],"o-",label=label,color=color)
    axes[1].set_xticks([1,2,3])
    axes[1].set_xlabel("Разбиение по авторам")
    axes[1].set_ylabel("Macro-F1 на внешних авторах")
    axes[1].set_title("Результаты по разбиениям",loc="left",weight="bold")
    axes[1].grid(axis="y",alpha=.2)
    axes[1].legend(frameon=False,fontsize=9)
    figure.suptitle("Kairos: смысл нового поста в контексте истории автора",x=.02,ha="left",weight="bold",fontsize=14)
    figure.text(.02,.02,"История строго раньше текущего поста; совпадающие времена исключены. Проверка внутри train, без нового официального test.",fontsize=9,color="#555555")
    figure.tight_layout(rect=[0,.065,1,.94])
    for extension in ("png","svg"):
        figure.savefig(OUT / f"history_results.{extension}",dpi=170,bbox_inches="tight",facecolor="white")
    plt.close(figure)
    print("Saved results/semantic_history/history_results.png and .svg")


if __name__ == "__main__":
    main()
