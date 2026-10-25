#!/usr/bin/env python3
"""Operational ETA sidecar; read progress only, never change a scientific job."""
import argparse
import json
import sqlite3
import statistics
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from kairos_lab.common import atomic_json

MSK = timezone(timedelta(hours=3))


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def task_progress(work, current_time, models=Path("/models")):
    status = read(work / "status.json")
    supervisor = (work / "supervisor.state").read_text().strip() if (work / "supervisor.state").exists() else ""
    if supervisor == "completed":
        return {"available": False, "reason": "Прогон завершён"}
    if (work / "STOP").exists() or supervisor in ("stopped", "deadline", "failed", "awaiting-refinement"):
        return {"available": False, "reason": supervisor or "STOP"}
    phase = supervisor.removeprefix("running ") or status.get("stage", "unknown")
    result = {"phase": phase, "available": False, "reason": "Нужны первые замеры скорости"}
    stage = status.get("stage")
    if stage == "train":
        progress = read(work / "training-status.json")
        candidate = progress.get("candidate", "")
        extension = status.get("extension")
        if extension and not candidate.startswith(f"refine-{extension}-"):
            result.update(task=f"Подготовка признаков {extension}",
                          reason="Новая серия готовит признаки; прежний fit не используется для ETA")
            return result
        job = progress.get("job", f"fold-{progress['fold']}" if "fold" in progress else "full")
        result["task"] = f"{candidate} / {job}"
        folder = work / "models" / candidate / "checkpoints" / job
        if progress.get("optimizer") == "pytorch":
            measured = read(work / "neural-progress.json")
            if measured.get("candidate") != candidate or measured.get("job") != job:
                return result
            if progress.get("at") and measured.get("at", "") < progress["at"]:
                return result
            steps, elapsed = measured.get("session_iterations", 0), measured.get("session_elapsed_seconds", 0)
            if steps > 0 and elapsed > 0:
                if current_time - (work / "neural-progress.json").stat().st_mtime > max(120, elapsed / steps * 200):
                    result["reason"] = "Для ETA нужен свежий замер PyTorch; прежнюю скорость не экстраполируем"
                    return result
                result.update(done=measured["iterations"], total=measured["total_iterations"], unit="steps",
                              per_unit=elapsed / steps, measured_at=(work / "neural-progress.json").stat().st_mtime,
                              basis="Фактическое время шагов PyTorch текущей сессии; checkpoint после каждой эпохи")
        elif candidate.startswith("tfidf-"):
            points = sorted((int(p.stem.split("-")[1]), p.stat().st_mtime)
                            for p in folder.glob("optimizer-*.joblib"))
            intervals = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(points, points[1:])
                         if b[0] > a[0] and b[1] > a[1]]
            if intervals:
                per_unit = statistics.median(intervals[-4:])
                done, measured_at = points[-1]
                if done < 400 and current_time - measured_at > 150 * per_unit:
                    result["reason"] = "Для ETA нужен свежий checkpoint; прежнюю скорость не экстраполируем"
                    return result
                result.update(done=done, total=400, unit="iterations", per_unit=per_unit, measured_at=measured_at,
                              basis="Медиана последних интервалов между checkpoint по 50 итераций")
        else:
            log = folder / "catboost_training.json"
            training = read(log)
            iterations = training.get("iterations", [])
            if iterations and iterations[-1].get("passed_time", 0) > 0:
                last = iterations[-1]
                done = last["iteration"] + 1
                result.update(done=done, total=training["meta"]["iteration_count"], unit="iterations",
                              per_unit=last["passed_time"] / done, measured_at=log.stat().st_mtime,
                              basis="Фактическое время итераций CatBoost")
    elif stage == "fetch-models":
        files = read(work / "environment/eta-model-files.json").get("files", {})
        started = read(work / "environment/runtime-fetch-models.json").get("at")
        if files and started:
            done = 0
            for name, size in files.items():
                path = models / name
                if not path.exists():
                    path = path.with_name(path.name + ".partial")
                done += min(size, path.stat().st_size) if path.exists() else 0
            if done:
                result.update(task="Загрузка весов", done=done, total=sum(files.values()), unit="bytes",
                              per_unit=(current_time - datetime.fromisoformat(started).timestamp()) / done,
                              measured_at=current_time, basis="Фактически полученные байты / время загрузки; размеры закреплённых файлов из metadata модели")
    elif stage in ("text", "images"):
        progress = read(work / "encoder-status.json")
        if progress.get("stage") != stage:
            return result
        result["task"] = "Текстовые эмбеддинги" if stage == "text" else "Визуальные эмбеддинги"
        if stage == "text" and progress.get("new_texts", 0) > 0:
            db_path = work / "cache/text.sqlite"
            try:
                with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2) as db:
                    done = db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
            except sqlite3.Error:
                return result
            result.update(done=done, total=progress["unique_texts"], unit="unique_texts",
                          per_unit=progress["elapsed_seconds"] / progress["new_texts"],
                          measured_at=(work / "encoder-status.json").stat().st_mtime,
                          basis="Скорость новых текстов; покрытие включает восстановленный SQLite-кэш")
        elif stage == "images" and progress.get("files_seen", 0) > 0:
            started = read(work / "environment/runtime-images.json").get("at")
            if started:
                measured_at = (work / "encoder-status.json").stat().st_mtime
                elapsed = measured_at - datetime.fromisoformat(started).timestamp()
                result.update(done=progress["files_seen"], total=progress["files_total"], unit="files",
                              per_unit=elapsed / progress["files_seen"], measured_at=measured_at,
                              basis="Фактическая скорость обработки файлов текущего запуска")
    if "per_unit" in result and result["per_unit"] > 0:
        age = max(0, current_time - result["measured_at"])
        per_unit = result.pop("per_unit")
        raw_remaining = max(0, result["total"] - result["done"]) * per_unit
        result.update(available=True, rate_per_minute=60 / per_unit,
                      remaining_seconds=round(max(0, raw_remaining - age)),
                      remaining_range_seconds=[round(max(0, raw_remaining * .8 - age)), round(max(0, raw_remaining * 1.25 - age) + 30)],
                      sample_age_seconds=round(age), confidence="measured")
        result["finish_at"] = datetime.fromtimestamp(current_time + result["remaining_seconds"], MSK).isoformat()
        if result["done"] >= result["total"]:
            result.update(fit_complete=True, remaining_seconds=None, remaining_range_seconds=None, finish_at=None)
        result.pop("reason", None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("/work"))
    args = parser.parse_args()
    work, current_time = args.work, time.time()
    current = task_progress(work, current_time)
    forecast = read(work / "environment/eta-forecast.json")
    selected = read(work / "selection.json")
    report = {"updated_at": datetime.fromtimestamp(current_time, MSK).isoformat(), "current_task": current,
              "pipeline_forecast": forecast or {"available": False, "reason": "Оставшиеся этапы ещё не измерены"},
              "macro_f1_oof": selected.get("metrics", {}).get("macro_f1"),
              "refinement_conditional": (work / "environment/refinement-request.json").exists(),
              "search_budget_ends": read(work / "search-budget.json").get("ends"),
              "note": "ETA не меняет лимиты. Оценка текущего fit не включает последующие этапы, запись результатов и дополнительные эксперименты."}
    atomic_json(work / "eta.json", report)
    lines = ["# Kairos — статус и ETA", "", f"Обновлено: **{report['updated_at']}** (МСК).", "",
             f"Текущая работа: **{current.get('task', current.get('phase', 'ожидание'))}**."]
    if current["available"]:
        if current["unit"] == "bytes":
            lines += [f"Прогресс: {current['done']/1024**3:.2f} / {current['total']/1024**3:.2f} ГиБ.",
                      f"Скорость: **{current['rate_per_minute']/60/1024**2:.2f} МиБ/с**."]
        else:
            lines += [f"Прогресс: {current['done']} / {current['total']} {current['unit']}.",
                      f"Скорость: **{current['rate_per_minute']:.1f} {current['unit']}/мин**."]
        if current.get("fit_complete"):
            lines += ["Обработка завершена; идут сохранение результатов и проверки."]
        else:
            low, high = current["remaining_range_seconds"]
            lines += [f"Осталось на эту работу: **примерно {round(current['remaining_seconds'] / 60)} мин** "
                  f"(диапазон {int(low / 60)}–{max(1, round(high / 60))} мин), окончание около **{current['finish_at'][11:16]} МСК**.",
                  f"Основание: {current['basis']}."]
    else:
        lines += [f"ETA текущей работы: {current['reason']}."]
    lines += ["", f"Текущий выбранный macro-F1 на author OOF: **{report['macro_f1_oof']}**.", "",
              "**До финального результата:** " + forecast.get("description", "Нужно измерить скорость следующих этапов."),
              "", "После основного прогона возможны дополнительные эксперименты при OOF < 0.70; их время учитывается отдельно.",
              "", "Обновляется при проверках монитора примерно каждые 15 минут. Последнее время обновления указано сверху."]
    tmp = work / "ETA.md.tmp"
    tmp.write_text("\n\n".join(lines) + "\n")
    tmp.replace(work / "ETA.md")
    (work / "logs").mkdir(exist_ok=True)
    with (work / "logs/eta.log").open("a") as log:
        log.write(json.dumps({"at": report["updated_at"], "task": current.get("task"),
                              "remaining_seconds": current.get("remaining_seconds"), "finish_at": current.get("finish_at")}) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
