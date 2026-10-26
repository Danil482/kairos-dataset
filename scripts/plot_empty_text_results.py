"""Export overall/empty-group tradeoffs and the missing-above-class diagnosis."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "results" / "empty_text"


def main():
    report = json.loads((OUT / "comparison.json").read_text())
    names = ["current","old_visual_route","group_calibration","guarded_history","empty_specialist","ocr_metadata","ocr_semantic","selected_route","balanced_route"]
    labels = ["Прежняя лучшая модель","Старая визуальная модель для пустых","Пороги для пустого текста","Защита от чужого контекста","Отдельная модель","+ признаки наличия OCR","+ смысл OCR-текста","Выбор готового варианта","Совместный подбор модели и порогов"]
    pairs = [(n,l) for n,l in zip(names,labels) if n in report["oof"]]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    figure,axes = plt.subplots(1,2,figsize=(14,6),gridspec_kw={"width_ratios":[1.35,1]})
    values = [report["slices"]["empty"]["models"][n]["macro_f1"] for n,_ in pairs]
    colors = ["#9ba6b3"]+["#4d948a"]*(len(pairs)-1)
    bars = axes[0].barh([l for _,l in pairs],values,color=colors)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0,max(values)+.08)
    axes[0].set_xlabel("Macro-F1: 5 726 постов без собственного текста")
    axes[0].set_title("Какие исправления помогают этой группе",loc="left",weight="bold")
    for bar,value in zip(bars,values):
        axes[0].text(value+.006,bar.get_y()+bar.get_height()/2,f"{value:.4f}",va="center")
    for i,(name,label) in enumerate(pairs):
        x = report["oof"][name]["macro_f1"]
        y = report["slices"]["empty"]["models"][name]["macro_f1"]
        axes[1].scatter([x],[y],s=65,color=colors[i],edgecolor="#285b58",zorder=3)
        axes[1].annotate(str(i+1),(x,y),xytext=(6,4),textcoords="offset points",fontsize=10)
    axes[1].axvline(report["oof"]["current"]["macro_f1"],linestyle="--",color="#9099a5",linewidth=1)
    axes[1].set_xlabel("Macro-F1 всей выборки")
    axes[1].set_ylabel("Macro-F1 постов без собственного текста")
    axes[1].set_title("Общее качество и качество группы",loc="left",weight="bold")
    axes[1].grid(alpha=.15)
    figure.suptitle("Kairos: почему падает качество без собственного текста",x=.02,ha="left",fontsize=14,weight="bold")
    figure.text(.02,.02,"Номера справа соответствуют строкам слева. Прогнозы постов с собственным текстом сохранены. Проверка внутри train; test не использован.",fontsize=9,color="#555555")
    figure.tight_layout(rect=[0,.065,1,.94])
    for extension in ("png","svg"):
        figure.savefig(OUT / f"empty_text_results.{extension}",dpi=170,bbox_inches="tight",facecolor="white")
    plt.close(figure)
    print("Saved results/empty_text/empty_text_results.png and .svg")


if __name__ == "__main__":
    main()
