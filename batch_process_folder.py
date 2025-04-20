import os
import json
import pandas as pd
from analysis.pipeline_test import run_pipeline

IMAGE_DIR = (r"C:\Code\data\poisoned\cifar10")
VALID_EXTENSIONS = [".jpg", ".jpeg", ".png"]

SAVE_EVERY = 100
JSON_PATH = "analysis_results.json"
CSV_PATH = "analysis_results.csv"

global_results = []

def collect_images(folder):
    images = []
    for root, _, files in os.walk(folder):
        for f in files:
            if any(f.lower().endswith(ext) for ext in VALID_EXTENSIONS):
                images.append(os.path.join(root, f))
    return images

def save_all_results():
    if not global_results:
        return

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(global_results, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON сохранён в {JSON_PATH}")

    rows = []
    for res in global_results:
        decision = res.get("decision", {})
        summary = decision.get("summary", {})

        row = {
            "image": res.get("image_path"),
            "suspicious": decision.get("suspicious", False),
            "triggered_count": len(decision.get("triggered_types", [])),
            "triggered_types": ";".join(decision.get("triggered_types", [])),
            "scale_factor": decision.get("scale_factor", -1),
            "meta_score": decision.get("meta_score", -1)
        }

        # Добавляем все метрики из summary
        if isinstance(summary, dict):
            for k, v in summary.items():
                if k not in row:
                    row[k] = v

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"📄 CSV сохранён в {CSV_PATH}")

def main():
    all_images = collect_images(IMAGE_DIR)
    print(f"🔍 Найдено изображений: {len(all_images)}\n")

    for i, path in enumerate(all_images):
        print(f"🔄 [{i+1}/{len(all_images)}] Обработка: {path}")
        try:
            result = run_pipeline(path, use_gpu=True, return_result=True)
            if result:
                global_results.append(result)

            if len(global_results) % SAVE_EVERY == 0:
                save_all_results()
        except Exception as e:
            print(f"❌ Ошибка при обработке {path}: {e}")

    save_all_results()

if __name__ == "__main__":
    main()
