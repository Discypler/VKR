# batch_process_folder.py

 import os
import json
import pandas as pd
from analysis.pipeline_test import run_pipeline

IMAGE_DIR = os.path.join("data", "poisoned", "cifar10")
VALID_EXTENSIONS = [".jpg", ".jpeg", ".png"]

SAVE_EVERY = 1000
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

    summary = []
    for res in global_results:
        row = {
            "image": res["image_path"],
            "suspicious": res["decision"].get("suspicious", False),
        }
        row.update(res["decision"].get("summary", {}))
        summary.append(row)

    df = pd.DataFrame(summary)
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
