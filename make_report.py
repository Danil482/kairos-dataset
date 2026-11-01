from __future__ import annotations

import argparse
from html import escape
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).parent / ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def main():
    ap = argparse.ArgumentParser(description="Собрать PDF по измеренным результатам ЛР 1")
    ap.add_argument("--results", type=Path, default=Path("results/lab1"))
    ap.add_argument("--output", type=Path, default=Path("submission/Lab1_Rostislav_Manzyuk.pdf"))
    ap.add_argument("--allow-fixture", action="store_true")
    args = ap.parse_args()
    path = args.results / "metrics.json"
    if not path.is_file():
        raise SystemExit("Нет измеренных результатов. Сначала запустите train_lab1.py.")
    m = json.loads(path.read_text())
    fixture = m.get("data_kind") != "official-release"
    if fixture and not args.allow_fixture:
        raise SystemExit("Отчёт для сдачи требует результатов на официальном датасете.")
    result = m["test"]["selected_blend"]
    baseline = m["test"]["official_baseline"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("Report", str(font_dir / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("Report-Bold", str(font_dir / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("Report", normal="Report", bold="Report-Bold")
    styles = getSampleStyleSheet()
    for name in ["Normal", "BodyText", "Title", "Heading1", "Heading2"]:
        styles[name].fontName = "Report-Bold" if name in ["Title", "Heading1", "Heading2"] else "Report"
    styles["BodyText"].fontSize = 9.5
    styles["BodyText"].leading = 13
    styles["BodyText"].spaceAfter = 6
    styles["Title"].fontSize = 19
    styles["Title"].leading = 25
    styles["Heading2"].fontSize = 12
    styles["Heading2"].keepWithNext = True
    styles["Heading2"].spaceBefore = 8
    story = []

    def paragraph(text, style="BodyText"):
        text = text.replace("—", "-").replace("–", "-")
        for char in ['«', '»', '“', '”']:
            text = text.replace(char, '"')
        story.append(Paragraph(text, styles[style]))

    def table(rows):
        t = Table(rows, hAlign="LEFT", repeatRows=1)
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Report"),
            ("FONTNAME", (0, 0), (-1, 0), "Report-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173f54")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 1), (-1, -1), .4, colors.lightgrey),
        ]))
        story.extend([t, Spacer(1, .3 * cm)])

    paragraph("ЛР 1. Как пост сработал относительно обычного результата автора", "Title")
    paragraph("Ростислав Манзюк, GitHub: grinchaus<br/>Social Media Analytics, Agents &amp; LLMs, ИТМО")
    if fixture:
        paragraph("ПРОВЕРКА НА ИСКУССТВЕННЫХ ДАННЫХ. НЕ РЕЗУЛЬТАТЫ ЛАБОРАТОРНОЙ.", "Heading1")
    paragraph("Что нужно было сделать", "Heading2")
    paragraph('Здесь задача - определить, получил ли LinkedIn-пост меньше реакций, чем автор обычно '
              'получает, примерно столько же или заметно больше. Классы - "below", "typical" и "above". '
              'Сравниваем пост с историей конкретного автора, поэтому большая аудитория сама по себе '
              'не означает успешный пост.')
    paragraph('Обычный уровень - медиана log(1 + reactions) по предыдущим собственным постам автора, '
              'которых должно быть хотя бы пять. Основная метрика - macro-F1: каждый из трёх классов '
              'получает одинаковый вес. Заодно смотрю QWK, accuracy и матрицу ошибок.')
    paragraph("Данные и проверка качества", "Heading2")
    paragraph(f"{'Для проверки использован искусственный набор' if fixture else 'Взял официальный релиз Kairos'}: "
              f"{m['train_posts']:,} постов от {m['train_authors']:,} авторов в train и "
              f"{m['test_posts']:,} постов от {m['test_authors']:,} авторов в test. "
              f"Внутри train отдельно оставил {m['validation_authors']:,} авторов для валидации "
              f"({m['validation_posts']:,} постов, seed {m['seed']}). Авторы между частями не пересекаются.")
    paragraph("Сначала проверил воспроизводимость разметки штатным kairos.label. "
              "Параметры моделей и смешивания подбирал только на авторах из train. "
              "Словарь TF-IDF, заполнение пропусков и масштабирование тоже обучаются внутри train. "
              "После выбора параметров зафиксировал конфигурацию, переобучил модели на всём train "
              "и посчитал итоговые результаты на официальном test.")
    paragraph("В своих признаках исправил разбор времени сбора: смешанные ISO-форматы "
              "с долями секунды и без них разбираются в UTC. Это восстановило 8 185 дат "
              "в размеченной части. Исходный baseline запускается с собственной предобработкой без изменений.")
    paragraph("Какие подходы попробовал", "Heading2")
    paragraph("Для начала воспроизвёл baseline - градиентный бустинг с балансировкой классов. "
              "Он использует длину и тип поста, хештеги, ссылки, время публикации и историю автора, "
              "но не читает слова в тексте.")
    paragraph("Дальше добавил CatBoost и расширил признаки. Посчитал средние и разброс реакций "
              "за последние 3, 5, 10 и 20 постов, отклонение недавних результатов от обычного уровня, "
              "частоту публикаций и изменения длины текста. Заодно добавил признаки оформления: "
              "вопросы, восклицания, абзацы, символы, упоминания и долю заглавных букв.")
    paragraph("Отдельно попробовал текстовую модель: TF-IDF по словам и парам слов, а также по "
              "символьным фрагментам длиной 3-5. К ним добавил числовые признаки и обучил "
              "логистическую регрессию с балансировкой классов. Так модель может использовать "
              "и содержание, и стиль поста.")
    paragraph("Заодно использовал вторую группу источников - профили авторов и метаданные вложений. "
              "Добавил навыки, описание профиля, отрасль, страну и язык, число мест работы и "
              "образований. Для вложений взял количество и размеры изображений, длительность "
              "видео, число страниц и извлечённый текст документов. Сами картинки и видео модель не читает.")
    paragraph("После этого проверил смешивание вероятностей CatBoost и текстовой модели. "
              "Вес моделей и поправки к классам выбрал по macro-F1 на валидации. "
              "По test ничего не подбирал. Валидационная оценка после такого выбора может быть "
              "оптимистичной, поэтому основной результат ниже - на отдельном test.")
    if m.get("embeddings"):
        encoder = m["embeddings"]
        paragraph("Отдельно проверил нейросетевое представление текста. Использовал "
                  "multilingual E5-small как фиксированный локальный encoder. Тексты постов и профилей "
                  "не передавались во внешний API. Перед кодированием добавлял префикс "
                  '"query: ", как рекомендуют авторы модели. Тексты длиннее 512 токенов обрезаются.')
        paragraph(f"PCA сжал векторы постов до {m['post_components']} признаков, "
                  f"а профилей - до {m['profile_components']}. PCA обучается только на обучающей "
                  "части, после чего признаки добавляются в CatBoost. Заодно добавил сходство "
                  "поста с профилем и предыдущими постами автора. Будущие тексты в эту историю не попадают.")
        if encoder.get("history_posts"):
            paragraph(f"Для текстовой истории дополнительно закодировал {encoder['history_posts']:,} "
                      "начальных постов. Это те пять постов каждого автора, на которых формируется "
                      "исходный уровень и для которых ещё нет целевой метки.")
        paragraph("Версия encoder зафиксирована: " + escape(encoder["revision"]) + ".")
        comparison = Path(__file__).parent / "submission" / "validation_experiments.json"
        if comparison.is_file():
            study = json.loads(comparison.read_text())
            if study.get("labels_sha256") == m["labels_sha256"]:
                paragraph(f"На той же валидации смешивание без E5 дало macro-F1 "
                          f"{study['without_embeddings_macro_f1']:.4f}, а выбранный вариант с E5 - "
                          f"{m['validation']['selected_blend']['macro_f1']:.4f}. "
                          "Увеличение размера представления и глубины дерева результата не улучшило. "
                          "Промежуточные опыты сохранены в validation_experiments.json.")
    paragraph("Что получилось", "Heading2")
    rows = [["Подход", "Val macro-F1", "Test macro-F1", "QWK", "Accuracy"]]
    names = {"official_baseline": "Baseline", "history_catboost": "CatBoost + E5" if m.get("embeddings") else "История + CatBoost",
             "text_and_metadata": "TF-IDF + признаки", "selected_blend": "Смешивание"}
    for key, label in names.items():
        v, t = m["validation"][key], m["test"][key]
        rows.append([label, f"{v['macro_f1']:.4f}", f"{t['macro_f1']:.4f}",
                     f"{t['qwk']:.4f}", f"{t['accuracy']:.4f}"])
    table(rows)
    paragraph("В README исходного репозитория указан baseline 0.5410. В таблице использую "
              f"результат его отдельного запуска в этом окружении - {baseline['macro_f1']:.4f}.")
    delta = result["macro_f1"] - baseline["macro_f1"]
    paragraph(f"Итоговый macro-F1 - <b>{result['macro_f1']:.4f}</b>. "
              f"Разница относительно воспроизведённого baseline - <b>{delta:+.4f}</b>. "
              "Обе модели оценены на одном и том же полном test.")
    if "confidence_intervals" in m:
        ci = m["confidence_intervals"]
        low, high = ci["macro_f1_95ci"]["selected_blend"]
        delta_low, delta_high = ci["selected_minus_baseline_95ci"]
        paragraph(f"95% доверительный интервал macro-F1 - [{low:.4f}; {high:.4f}]. "
                  f"Интервал разницы с baseline - [{delta_low:+.4f}; {delta_high:+.4f}]. "
                  f"Посчитал парным bootstrap по авторам, {ci['samples']} повторов. "
                  "Посты одного автора при этом остаются вместе.")
    cfg = m["selected_config"]
    paragraph(f"В итоговой конфигурации вес CatBoost по классам - {cfg['tree_weight']}, "
              f"множители вероятностей классов - {escape(str(cfg['class_weights']))}, "
              f"число итераций CatBoost - {m['catboost_iterations']}.")
    paragraph(f"Глубина дерева - {m.get('catboost_depth', 6)}, "
              f"C логистической регрессии - {m.get('text_c', 1.0)}.")
    per = result["classification_report"]
    table([["Класс", "Precision", "Recall", "F1", "Постов"]] + [
        [key, f"{per[key]['precision']:.4f}", f"{per[key]['recall']:.4f}",
         f"{per[key]['f1-score']:.4f}", str(int(per[key]['support']))]
        for key in ["below", "typical", "above"]])
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    matrix = np.array(result["confusion_matrix"])
    ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=["below", "typical", "above"],
           yticklabels=["below", "typical", "above"], xlabel="Предсказанный класс", ylabel="Настоящий класс")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:,}", ha="center", va="center",
                color="white" if value > matrix.max() / 2 else "black")
    fig.tight_layout()
    image = args.results / "confusion_matrix.png"
    fig.savefig(image, dpi=160)
    plt.close(fig)
    story.append(Image(str(image), width=11 * cm, height=7.6 * cm))
    paragraph("Почему это может работать и где ограничения", "Heading2")
    paragraph("История показывает, насколько стабилен автор и не изменился ли его обычный результат "
              "в последнее время. Текстовая модель добавляет информацию, которой baseline не пользуется. "
              "Сравнение моделей показывает полезность целых подходов, но не доказывает вклад каждого "
              "отдельного признака - для этого нужна более подробная абляция.")
    paragraph("Реакции текущего поста, его residual, label и идентификаторы автора и поста "
              "не попадают в матрицу признаков. Окна истории сдвинуты на один пост назад. "
              "Проверки меняют текущие и будущие реакции и подтверждают, что признаки текущего "
              "и более ранних постов остаются прежними. Коды людей и компаний в тексте нормализованы.")
    paragraph("Есть ограничение самого набора: реакции - это снимок на момент сбора данных, "
              "а число подписчиков измерено позже публикации. Возраст поста на момент сбора "
              "тоже используется как признак, хотя он связан с процедурой сбора данных. "
              "Поэтому результат относится к этому офлайн-заданию и не означает, что столько же "
              "получится при прогнозе сразу в момент публикации.")
    paragraph("Профиль автора тоже является снимком на момент сбора. Его содержание может "
              "отличаться от того, что было в профиле при публикации старых постов. "
              "Это отдельное ограничение использования профиля как второго источника.")
    paragraph("Историю test-автора использую так же, как официальный baseline: более ранние посты "
              "с известными реакциями могут участвовать в признаках более поздних. Картинки и видео "
              "в этой версии не использовал.")
    paragraph("Как воспроизвести", "Heading2")
    paragraph(f"Python {escape(m['python'])}, scikit-learn {escape(m['sklearn'])}. "
              f"Обучение и оценка моделей заняли {m['elapsed_seconds']:.0f} секунд. "
              "Кодирование текстов - отдельный подготовительный шаг. "
              "После установки requirements-embeddings.txt и распаковки данных достаточно запустить:")
    paragraph("<font face='Courier'>python run_lab1.py --data data</font>")
    paragraph("Команда проверяет разметку, готовит локальные эмбеддинги, обучает модели "
              "с зафиксированными параметрами из selection.json, проверяет предсказания "
              "и собирает комплект для сдачи.")
    paragraph("В репозитории лежат notebook, предсказания и метрики. "
              "Обученную модель можно сохранить локально и проверить повторный inference "
              "без переобучения. Данные и пароли в репозиторий не загружены.")
    paragraph('Материалы: <a href="https://github.com/Danil482/kairos-dataset">официальный baseline</a>, '
              '<a href="https://kairos-labs.ai/dataset/">датасет Kairos</a>, '
              'слайды и уточнения по сдаче из чата курса.')
    if m.get("embeddings"):
        paragraph('Encoder: <a href="https://huggingface.co/intfloat/multilingual-e5-small">'
                  'intfloat/multilingual-e5-small, описание модели и правила кодирования</a>.')

    def footer(canvas, doc):
        canvas.setFont("Report", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(1.7 * cm, 1 * cm, "Ростислав Манзюк | ЛР 1")
        canvas.drawRightString(19.3 * cm, 1 * cm, str(doc.page))

    SimpleDocTemplate(str(args.output), pagesize=(21 * cm, 29.7 * cm),
                      title="ЛР 1 - Прогноз результата поста относительно автора",
                      author="Ростислав Манзюк",
                      subject="Social Media Analytics, Agents & LLMs - ITMO University",
                      leftMargin=1.7 * cm, rightMargin=1.7 * cm,
                      topMargin=1.5 * cm, bottomMargin=1.5 * cm).build(
                          story, onFirstPage=footer, onLaterPages=footer)
    print(args.output)


if __name__ == "__main__":
    main()
