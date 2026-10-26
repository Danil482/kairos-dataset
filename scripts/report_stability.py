"""Reproduce saved predictions and produce the stability research report/figure."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

from scripts.stability_models import OUT,PRIORITY,FAMILIES,SEEDS,RULES,NAMES,MEANS,INDIVIDUALS,bootstrap_difference
from scripts.semantic_embeddings import save_json,fingerprint
from train_maria import metrics

FAMILY_LABELS = dict(quote="История цитирования",ordinal="Порядковая модель",joint="Порядковая + цитирование")
RULE_LABELS = dict(grid="прежняя сетка",fine_grid="плотная сетка",bootstrap_grid="bootstrap прежней сетки",bootstrap_fine="bootstrap плотной сетки")


def label(name):
    if name=="v2":
        return "Кандидат v2"
    if name=="old_ordinal":
        return "Прежняя порядковая, один seed"
    family,rule = name.split("_mean_")
    return FAMILY_LABELS[family]+" — "+RULE_LABELS[rule]


def number(x):
    return f"{x:.4f}".replace(".",",")


def interval(value):
    return "["+"; ".join(f"{value[key]:+.5f}".replace(".",",") for key in ("low","high"))+"]"


def audit_and_diagnose(result):
    protocol = json.loads((OUT / "protocol.json").read_text())
    if not result["complete"] or result["config"]!=protocol:
        raise ValueError("Incomplete or inconsistent comparison")
    if protocol["priority_features_sha256"]!=fingerprint(PRIORITY / "features.joblib"):
        raise ValueError("Feature cache changed")
    parts,artifacts,selection = [],{},{}
    quality_probabilities = {name:[] for name in INDIVIDUALS+[f"{family}_mean_grid" for family in FAMILIES]}
    for number in (1,2,3):
        directory = OUT / f"fold_{number}"
        report = json.loads((directory / "report.json").read_text())
        if not report["complete"] or report["config"]!=protocol or set(report["models"])!=set(NAMES):
            raise ValueError("Fold protocol incomplete/changed")
        for name in ("semantic_pca.joblib","visual_pca.joblib"):
            original = ROOT / "results" / "semantic_history" / f"fold_{number}" / name
            frozen = ROOT / "results" / "candidates" / "v2" / "results" / "semantic_history" / f"fold_{number}" / name
            if fingerprint(directory / name)!=fingerprint(original) or fingerprint(original)!=fingerprint(frozen):
                raise ValueError("PCA differs from the frozen v2 representation")
        author_sets = [set(report["author_plan"][key]) for key in ("fitting_authors","calibration_authors","validation_authors")]
        if any(author_sets[i]&author_sets[j] for i in range(3) for j in range(i)):
            raise ValueError("Author leakage")
        cal = pd.read_csv(directory / "calibration_keys.csv")
        val = pd.read_csv(directory / "predictions.csv")
        if set(cal.author)!=author_sets[1] or set(val.author)!=author_sets[2]:
            raise ValueError("Prediction populations changed")
        stored = {}
        for name in NAMES:
            p = np.load(directory / f"{name}_probabilities.npz")
            for key,table in (("calibration",cal),("validation",val)):
                array = p[key]
                if array.shape!=(len(table),3) or not np.isfinite(array).all() or (array<0).any() or not np.allclose(array.sum(axis=1),1):
                    raise ValueError("Invalid saved probabilities")
            decision = report["models"][name]["calibration"]
            np.testing.assert_array_equal((p["validation"]*decision["multipliers"]).argmax(axis=1),val[name])
            if metrics(val.label,val[name])!=report["models"][name]["metrics"]:
                raise ValueError("Metrics do not reproduce")
            details = report["models"][name]["details"]
            if name in INDIVIDUALS:
                if not details["roundtrip_passed"] or details["script_sha256_at_fit"]!=fingerprint(ROOT / "scripts" / "stability_models.py"):
                    raise ValueError("Classifier provenance changed")
                if details["reused"]:
                    for file,expected in details["sources"].items():
                        if fingerprint(ROOT / file)!=expected:
                            raise ValueError("Reused artifact changed")
                else:
                    schema = json.loads((directory / f"{name}_schema.json").read_text())
                    if schema["config"]!=protocol or schema["parameters"]["random_seed"]!=details["seed"]:
                        raise ValueError("Schema/model settings differ")
                    if set(schema["columns"])&{"author","post","label","split","reactions","resid"}:
                        raise ValueError("Forbidden model column")
            stored[name] = {key:p[key] for key in ("calibration","validation")}
        for family in FAMILIES:
            for key in ("calibration","validation"):
                mean = np.mean([stored[f"{family}_s{i}"][key] for i in range(3)],axis=0)
                mean /= mean.sum(axis=1,keepdims=True)
                for rule in RULES:
                    np.testing.assert_allclose(stored[f"{family}_mean_{rule}"][key],mean,rtol=1e-12,atol=1e-12)
        for name in quality_probabilities:
            quality_probabilities[name].append(stored[name]["validation"])
        chosen = max(MEANS,key=lambda name:report["models"][name]["calibration"]["calibration_macro_f1"])
        val["inner_selected_mean"] = val[chosen]
        selection[str(number)] = dict(chosen=chosen,calibration_macro_f1=report["models"][chosen]["calibration"]["calibration_macro_f1"])
        parts.append(val.assign(fold=number))
        artifacts[str(number)] = {file.name:fingerprint(file) for file in sorted(directory.glob("*.cbm"))}
    rebuilt = pd.concat(parts,ignore_index=True)
    oof = pd.read_csv(OUT / "oof_predictions.csv")
    pd.testing.assert_frame_equal(oof[["author","post","label","fold"]+NAMES],rebuilt[["author","post","label","fold"]+NAMES])
    diagnostic = dict(scope="Exploratory selection diagnostic among the 12 seed-mean/rule variants; labels for selection are inner only",
        selection=selection,metrics=metrics(rebuilt.label,rebuilt.inner_selected_mean),
        versus_old_ordinal_95=bootstrap_difference(rebuilt.label,rebuilt.inner_selected_mean,oof.old_ordinal,rebuilt.author),
        joint_bootstrap_fine_minus_fine_95=bootstrap_difference(rebuilt.label,rebuilt.joint_mean_bootstrap_fine,rebuilt.joint_mean_fine_grid,rebuilt.author))
    truth = np.eye(3)[rebuilt.label.to_numpy()]
    diagnostic["probability_quality"] = {}
    for name,arrays in quality_probabilities.items():
        probability = np.concatenate(arrays)
        diagnostic["probability_quality"][name] = dict(log_loss=float(log_loss(rebuilt.label,probability,labels=[0,1,2])),
            multiclass_brier=float(np.mean(np.sum((probability-truth)**2,axis=1))),
            raw_argmax_macro_f1=metrics(rebuilt.label,probability.argmax(axis=1))["macro_f1"])
    rebuilt[["author","post","label","fold","inner_selected_mean"]].to_csv(OUT / "inner_selected_predictions.csv",index=False)
    save_json(OUT / "diagnostics.json",diagnostic)
    save_json(OUT / "artifact_audit.json",dict(complete=True,folds=3,variants_per_fold=len(NAMES),
        predictions_reproduced=True,metrics_reproduced=True,author_sets_disjoint=True,equal_seed_weights_verified=True,
        model_hashes=artifacts,source_hashes={name:fingerprint(ROOT / "scripts" / name) for name in ("stability_models.py","report_stability.py")}))
    return diagnostic


def plot(result):
    fig,axes = plt.subplots(1,2,figsize=(14,6.2),gridspec_kw=dict(width_ratios=[1,1.2]))
    colors = ["#236a9f","#26957e","#c17c2b"]
    for position,family in enumerate(FAMILIES):
        scores = result["seed_spread"][family]["scores"]
        for i,value in enumerate(scores):
            axes[0].scatter(position+(i-1)*.08,value,color=colors[position],s=55)
            axes[0].annotate(str(i+1),(position+(i-1)*.08,value),xytext=(4,3),textcoords="offset points",fontsize=8)
        mean = result["oof"][f"{family}_mean_grid"]["macro_f1"]
        axes[0].scatter(position,mean,marker="D",s=90,color=colors[position],edgecolors="black",zorder=4)
    axes[0].set_xticks(range(3),["Цитирование","Порядковая","Вместе"])
    axes[0].set_xlim(-.4,2.4)
    axes[0].set_title("Три seed и усреднение прогнозов")
    axes[0].set_ylabel("OOF macro-F1")
    axes[0].axhline(result["oof"]["old_ordinal"]["macro_f1"],color="#586779",linestyle="--",linewidth=1)
    for family,color in zip(FAMILIES,colors):
        values = [result["oof"][f"{family}_mean_{rule}"]["macro_f1"] for rule in RULES]
        axes[1].plot(range(4),values,"o-",color=color,label=FAMILY_LABELS[family])
    axes[1].set_xticks(range(4),["Прежняя\nсетка","Плотная\nсетка","Bootstrap\nпрежней","Bootstrap\nплотной"])
    axes[1].set_title("Правило выбора класса для среднего прогнозов")
    axes[1].axhline(result["oof"]["old_ordinal"]["macro_f1"],color="#586779",linestyle="--",linewidth=1)
    axes[1].legend(frameon=False,fontsize=9,loc="best")
    for ax in axes:
        ax.grid(axis="y",alpha=.2)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Устойчивость улучшений: случайные инициализации и правила выбора",fontsize=15)
    fig.text(.5,.015,"71 395 постов · те же 3 авторских разбиения train · ромб — среднее вероятностей · пунктир — прежние 0,6630",ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.94])
    for ext in ("png","svg"):
        fig.savefig(OUT / f"stability_results.{ext}",dpi=170,bbox_inches="tight")
    plt.close(fig)


def write_report(result,diagnostic):
    best = max(MEANS,key=lambda name:result["oof"][name]["by_class"]["macro avg"]["f1-score"])
    lines = ["## Устойчивость по seed, совместное использование признаков и решающее правило — 26 октября 2026","",
        "Проверены три семейства: CatBoost с историей того же режима цитирования; два порядковых CatBoost на прежних признаках v2; порядковая модель с добавлением истории цитирования. Использованы **три заранее фиксированных seed: 20261025, 20261026, 20261027**. Для каждого семейства усреднялись вероятности с равными весами 1/3. Лучший seed по внешним меткам не выбирался.","",
        "Обучены **36 новых классификаторов**, ещё девять готовых классификаторов первого seed переиспользованы с проверкой параметров, признаков, порядка строк и контрольных сумм. В каждом разбиении сохранены 9 отдельных прогнозов и 12 вариантов усреднения/решающего правила. Сравнение проводится на прежних 71 395 постах и 3 295 авторах официального train. Test и обучение финальной модели на всём train не выполнялись.","",
        "### Что менялось и что фиксировалось","",
        "Бюджет прежний: до 800 деревьев, глубина 6, learning rate 0,045, L2=5, ранняя остановка после 100 итераций без улучшения на внутренних calibration-авторах. Преобразования PCA переиспользованы из соответствующих fitting-групп. E5, SigLIP и исходные признаки не пересчитывались. Новые seed меняют только случайность обучения CatBoost; они не являются новыми разбиениями данных.","",
        "У среднего трёх прогнозов сравниваются четыре правила. Прежняя сетка содержит 144 пары множителей below/above при typical=1. Плотная сетка сохраняет прежние значения и добавляет геометрические середины интервалов — 529 пар в тех же пределах 0,25–8. Для каждой сетки дополнительно выбирается лучшее правило на 100 bootstrap-выборках **внутренних авторов**, затем множители усредняются геометрически. Все посты выбранного автора входят в перевыборку вместе. Это предложенная проверка устойчивости решающего правила, не стандартная вероятностная калибровка Platt/isotonic.","",
        "Внешние авторы не участвуют в выборе деревьев, множителей или весов seed. Внутренняя calibration-группа по прежнему протоколу используется и для остановки, и для решающего правила: её собственные метрики не являются независимой оценкой качества. Первичное сравнение этого этапа заранее задано как совместная модель против порядковой при одинаковом усреднении трёх seed и прежней сетке.","",
        "### Разброс одиночных запусков","","| Семейство | Seed 1 | Seed 2 | Seed 3 | Среднее метрик | SD метрик |","|---|---:|---:|---:|---:|---:|"]
    for family in FAMILIES:
        spread = result["seed_spread"][family]
        lines.append("| "+FAMILY_LABELS[family]+" | "+" | ".join(number(v) for v in spread["scores"]+[spread["mean"],spread["std"]])+" |")
    lines += ["","Разброс одиночных запусков включает и изменение модели, и повторный выбор её множителей на внутренних авторах. Среднее метрик отдельных seed отличается от метрики среднего вероятностей; ниже приведён именно второй вариант.","",
        "### Усреднение прогнозов и решающие правила","","| Вариант | Macro-F1 всех постов | Без своего текста | История < 10 | QWK |","|---|---:|---:|---:|---:|"]
    for name in ["v2","old_ordinal"]+MEANS:
        score = result["oof"][name]
        lines.append(f"| {label(name)} | {number(score['macro_f1'])} | {number(result['slices']['empty_original']['macro_f1'][name])} | {number(result['slices']['short_history']['macro_f1'][name])} | {number(score['qwk'])} |")
    lines += ["",f"Лучшее наблюдавшееся усреднение этого этапа: **{label(best)} — {number(result['oof'][best]['macro_f1'])}**. Это результат разработки, а не новая оценка на официальном test.","",
        "Первичное парное сравнение совместной модели с порядковой (три seed, прежняя сетка): "+interval(result["comparisons_95"]["joint_mean_grid_minus_ordinal_mean_grid"])+".","",
        "Интервалы изменений относительно прежней порядковой модели одного seed (0,6630):",""]
    for name in MEANS:
        lines.append("- "+label(name)+": "+interval(result["comparisons_95"][name+"_minus_old_ordinal"])+".")
    lines += ["","Диагностический выбор среди усреднённых вариантов только по внутренней метрике дал **"+number(diagnostic["metrics"]["macro_f1"])+"**. Выбраны "+", ".join(f"fold {k}: {v['chosen']}" for k,v in diagnostic["selection"].items())+". Интервал его разницы с прежней порядковой моделью: "+interval(diagnostic["versus_old_ordinal_95"])+".","",
        "### Вероятности отдельно от решающего правила","",
        "| Среднее трёх seed | Log loss ↓ | Multiclass Brier ↓ | Macro-F1 обычного argmax |","|---|---:|---:|---:|"]
    for family in FAMILIES:
        quality = diagnostic["probability_quality"][f"{family}_mean_grid"]
        lines.append("| "+FAMILY_LABELS[family]+" | "+" | ".join(number(quality[key]) for key in ("log_loss","multiclass_brier","raw_argmax_macro_f1"))+" |")
    lines += ["","У четырёх правил одного семейства вероятности одинаковы — это проверено по сохранённым файлам. Поэтому разница их macro-F1 относится к выбору класса, а не к обучению новых текстовых или визуальных признаков. Значения одиночных seed сохранены в `diagnostics.json`.","",
        "### Что дало прирост и где продолжать","",
        "Лучший наблюдавшийся вариант — совместная порядковая модель с историей цитирования, усреднение трёх seed и bootstrap плотной сетки: **0,6665 против прежних 0,6630**. У пустых оригиналов macro-F1 вырос с 0,4755 до 0,4809, у коротких историй — с 0,6488 до 0,6511. Это улучшение общей модели, но проблема пустых текстов остаётся: отдельные ранее проверенные правила для этой группы давали больше за счёт общей метрики.","",
        "Вклад решающего правила виден в контролируемом сравнении на одних и тех же вероятностях: у совместной модели **0,6642 → 0,6665**. Интервал этой разницы — "+interval(result["comparisons_95"]["joint_mean_bootstrap_fine_minus_joint_mean_grid"])+". По сравнению с одной лишь плотной сеткой (0,6648) интервал изменения при bootstrap составляет "+interval(diagnostic["joint_bootstrap_fine_minus_fine_95"])+". На втором fold плотная сетка дала слабый результат, несмотря на высокий внутренний score; усреднение правил уменьшило этот провал.","",
        "Отдельный вклад истории цитирования в macro-F1 пока менее убедителен: первичное сравнение совместного и порядкового усреднений с прежней сеткой имеет интервал, включающий ноль. При этом совместный вариант показал меньшие log loss и Brier и больший macro-F1 обычного argmax. Это поддерживает дальнейшую проверку его вероятностей, но не доказывает причинный эффект конкретного признака.","",
        "Следующий приоритет — независимее разделить внутренние группы для ранней остановки и выбора решающего правила, проверить небольшую сетку регуляризации на этих группах и устойчивость новых результатов. Следующий содержательный опыт — добавить к порядковой модели раздельные цитату/документ при том же бюджете, затем проверить, сколько смысла теряется при сжатии E5 до 64 компонент. Это ещё не выполненные эксперименты. Усиление глобального качества отдельно сопоставлять с результатами пустых текстов; выбирать решение только по наибольшему числу в development-таблице недостаточно.","",
        "### Границы выводов и воспроизведение","",
        "500 парных bootstrap-перевыборок внешних авторов дают условные интервалы для сохранённых прогнозов. Они не включают многократный выбор методов на тех же development-разбиениях, подбор семейства по итоговой таблице и весь разброс оптимизации. Три seed — небольшая проверка, а не доказательство универсальной устойчивости. Поправка на множественные сравнения не применялась. Bootstrap внутренних авторов для решения и bootstrap внешних авторов для интервалов — разные процедуры.","",
        "Все сохранённые прогнозы восстановлены из вероятностей и множителей, метрики пересчитаны; проверены непересечение авторов и равные веса трёх seed. Новые классификаторы повторно загружены и проверены на воспроизведение вероятностей. Прошли **42 теста**, включая воспроизведение прежнего подбора множителей, группировку по авторам и устойчивость расчёта к порядку строк. Ускоренный расчёт интервалов в точности совпал с прежним авторским bootstrap в отдельной проверке.","",
        "Артефакты: `results/stability_research/protocol.json`, `comparison.json`, `diagnostics.json`, `artifact_audit.json`, модели в `fold_*/` и `stability_results.png`. Запуск/продолжение — `train_stability_research.cmd`; статус — `check_status.cmd`. Прежние результаты и кандидат v2 сохранены.","",
        "Источники методических принципов: [параметры CatBoost](https://catboost.ai/docs/en/references/training-parameters/common) и [разделение обучения вероятностей и настройки решающего порога](https://scikit-learn.org/stable/modules/classification_threshold.html). Конкретный многоклассовый bootstrap множителей реализован для этого эксперимента; это не использование `TunedThresholdClassifierCV`.",""]
    path = ROOT / "IMPROVEMENT_PLAN_RU.md"
    content = path.read_text(encoding="utf-8")
    heading = lines[0]
    if heading in content:
        start = content.index(heading)
        stop = content.find("\n## ",start+len(heading))
        content = content[:start]+"\n".join(lines)+"\n"+content[stop+1:] if stop>=0 else content[:start]+"\n".join(lines)
    else:
        position = content.index("\n## ")
        content = content[:position]+"\n"+"\n".join(lines)+content[position:]
    paragraphs = content.split("\n\n")
    if paragraphs[1].startswith("Проверка проведена на"):
        paragraphs[1] = ("Проверка проведена на 71 395 постах 3 295 авторов официального train. Лучшее усреднение нового этапа — **OOF macro-F1 "+number(result["oof"][best]["macro_f1"])+"**, **"+label(best)+"**. Предыдущие ориентиры: кандидат v2 — 0,6605, порядковая модель одного seed — 0,6630. Все числа разработки получены на прежних трёх разбиениях по авторам внутри train. Сохранённая первая модель имеет официальный test-результат 0,6229; новые этапы на официальном test не оценивались.")
        content = "\n\n".join(paragraphs)
    path.write_text(content,encoding="utf-8")
    return best


def main():
    result = json.loads((OUT / "comparison.json").read_text())
    diagnostic = audit_and_diagnose(result)
    plot(result)
    best = write_report(result,diagnostic)
    print(json.dumps(dict(audit="passed",best_mean=best,macro_f1=result["oof"][best]["macro_f1"],
        selected_inner_macro_f1=diagnostic["metrics"]["macro_f1"]),indent=2))


if __name__ == "__main__":
    main()
