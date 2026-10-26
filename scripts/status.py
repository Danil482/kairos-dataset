"""Show local dataset progress without opening the private data."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
print(f"Post files ready: {len(list((root / 'data' / 'posts').glob('AUTHOR_*.json')))}")
print(f"Labels and split ready: {(root / 'data' / 'labels.csv').is_file() and (root / 'data' / 'split.json').is_file()}")
manifest = root / "archives" / "manifest.json"
if manifest.exists():
    volumes = json.loads(manifest.read_text())["volumes"]
    total = sum(v["size"] for v in volumes)
    downloaded = 0
    verified = 0
    for v in volumes:
        complete = root / "archives" / v["name"]
        partial = complete.with_name(complete.name + ".part")
        if complete.exists():
            downloaded += complete.stat().st_size
            verified += 1
        elif partial.exists():
            downloaded += partial.stat().st_size
    print(f"Downloaded: {downloaded / 1e9:.2f} / {total / 1e9:.2f} GB ({100 * downloaded / total:.1f}%)")
    print(f"Completed volumes: {verified} / {len(volumes)}")
for kind in ("core", "video"):
    print(f"{kind} extracted and verified: {(root / '.local' / f'verified_{kind}.json').exists()}")
status = root / ".local" / "setup_status.json"
if status.exists():
    print(status.read_text())
for filename in ("baseline.json", "maria/test_metrics.json"):
    result = root / "results" / filename
    if result.exists():
        print(f"Results available: {result}")

training = root / "results" / "lecture_models"
if training.exists():
    for number in (1, 2, 3):
        report = training / f"fold_{number}" / "report.json"
        if report.exists():
            try:
                saved = json.loads(report.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                print(f"Lecture fold {number}: report is being updated; check again")
                continue
            state = "complete" if saved.get("complete") else "partial (see training log)"
            print(f"Lecture fold {number}: {len(saved['models'])}/5 model results saved, {state}")
    comparison = training / "comparison.json"
    if comparison.exists():
        print(f"Lecture model comparison available: {comparison}")
    calibrated = training / "decision_audit" / "comparison.json"
    if calibrated.exists():
        print(f"Final lecture decision audit available: {calibrated}")

semantic = root / "results" / "semantic"
if (semantic / "embedding_manifest.json").exists():
    print("Semantic E5 vectors: complete")
elif (semantic / "embedding_progress.json").exists():
    progress = json.loads((semantic / "embedding_progress.json").read_text())
    print(f"Semantic E5 vectors saved: {progress['completed_chunks']} / {progress['total_chunks']} chunks")
for number in (1, 2, 3):
    report = semantic / f"fold_{number}" / "report.json"
    if report.exists():
        saved = json.loads(report.read_text())
        print(f"Semantic fold {number}: {len(saved['models'])}/5 model results saved, complete={saved.get('complete', False)}")
if (semantic / "comparison.json").exists():
    print(f"Semantic comparison available: {semantic / 'comparison.json'}")

visual = root / "results" / "visual"
if (visual / "embedding_manifest.json").exists():
    print("Visual SigLIP 2 image vectors: complete")
elif (visual / "embedding_progress.json").exists():
    progress = json.loads((visual / "embedding_progress.json").read_text())
    print(f"Visual SigLIP 2 images saved: {progress['completed_images']} / {progress['total_images']}")
for number in (1, 2, 3):
    report = visual / f"fold_{number}" / "report.json"
    if report.exists():
        saved = json.loads(report.read_text())
        print(f"Visual fold {number}: {len(saved['models'])}/4 model results saved, complete={saved.get('complete', False)}")
if (visual / "comparison.json").exists():
    print(f"Visual comparison available: {visual / 'comparison.json'}")

history = root / "results" / "semantic_history"
if (history / "features_manifest.json").exists():
    saved = json.loads((history / "features_manifest.json").read_text())
    print(f"Semantic author history: {len(saved['novelty_columns'])} novelty + {len(saved['neighbor_columns'])} neighbor features ready")
elif (history / "embedding_progress.json").exists():
    saved = json.loads((history / "embedding_progress.json").read_text())
    print(f"Additional history E5 chunks: {saved['completed_chunks']} / {saved['total_chunks']}")
for number in (1, 2, 3):
    path = history / f"fold_{number}" / "report.json"
    if path.exists():
        saved = json.loads(path.read_text())
        print(f"Semantic history fold {number}: {len(saved['models'])}/4 models saved; text_complete={saved.get('text_complete', False)}, complete={saved.get('complete', False)}")
if (history / "comparison.json").exists():
    print(f"Semantic history comparison available: {history / 'comparison.json'}")

empty = root / "results" / "empty_text"
ocr = empty / "ocr"
if (ocr / "manifest.json").exists():
    saved = json.loads((ocr / "manifest.json").read_text())
    print(f"Empty-post OCR complete: {saved['images']} images, {saved['posts_with_text']} posts with text")
elif (ocr / "progress.json").exists():
    saved = json.loads((ocr / "progress.json").read_text())
    print(f"Empty-post OCR images saved: {saved['completed']} / {saved['total']}")
for number in (1, 2, 3):
    path = empty / f"fold_{number}" / "report.json"
    if path.exists():
        saved = json.loads(path.read_text())
        print(f"Empty-post fold {number}: {len(saved['models'])}/8 variants saved, complete={saved.get('complete', False)}")
if (empty / "comparison.json").exists():
    print(f"Empty-post comparison available: {empty / 'comparison.json'}")

priority = root / "results" / "priority_research"
if (root / "results" / "candidates" / "v2" / "manifest.json").exists():
    print("Candidate v2: frozen three-fold snapshot available")
if (priority / "features_manifest.json").exists():
    print("Priority research: six feature blocks ready")
elif (priority / "e5" / "embedding_progress.json").exists():
    saved = json.loads((priority / "e5" / "embedding_progress.json").read_text())
    print(f"Separate-field E5 chunks: {saved['completed_chunks']} / {saved['total_chunks']}")
for number in (1, 2, 3):
    path = priority / f"fold_{number}" / "report.json"
    if path.exists():
        saved = json.loads(path.read_text())
        print(f"Priority fold {number}: {len(saved['models'])}/12 variants saved, complete={saved.get('complete', False)}")
if (priority / "comparison.json").exists():
    print(f"Priority research comparison available: {priority / 'comparison.json'}")

stability = root / "results" / "stability_research"
for number in (1, 2, 3):
    path = stability / f"fold_{number}" / "report.json"
    if path.exists():
        saved = json.loads(path.read_text())
        print(f"Stability fold {number}: {len(saved['models'])}/21 variants saved, complete={saved.get('complete', False)}")
if (stability / "comparison.json").exists():
    print(f"Stability research comparison available: {stability / 'comparison.json'}")

separated = root / "results" / "separated_research"
for number in (1, 2, 3):
    path = separated / f"fold_{number}" / "report.json"
    if path.exists():
        saved = json.loads(path.read_text())
        print(f"Separated-role fold {number}: {len(saved['models'])}/12 forecasts, {2*len(saved['fits'])} classifiers saved, complete={saved.get('complete', False)}")
if (separated / "comparison.json").exists():
    print(f"Separated-role comparison available: {separated / 'comparison.json'}")

official = root / "results" / "official_test_v3"
if (official / "test_metrics.json").exists():
    saved = json.loads((official / "test_metrics.json").read_text())
    print(f"Official test complete: macro-F1={saved['macro_f1']}, QWK={saved['qwk']}, posts={saved['rows']}")
elif (official / "predictions.csv").exists():
    print("Official test predictions ready; metrics not yet evaluated")
elif (official / "visual" / "embedding_progress.json").exists():
    saved = json.loads((official / "visual" / "embedding_progress.json").read_text())
    print(f"Official test images: {saved['completed_images']}/{saved['total_images']}")
