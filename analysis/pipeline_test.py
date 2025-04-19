import sys
import cv2
import json
import numpy as np
from PIL import Image
from analysis.all_in_one_analyzer import analyze_all
from analysis.decision_maker import decide_from_anomalies

def run_pipeline(image_path, use_gpu=True, save_json=True):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"❌ Невозможно загрузить изображение: {image_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    height, width = img_rgb.shape[:2]
    short_side = min(width, height)
    scale_factor = short_side / 64.0

    anomalies = analyze_all(img_rgb, use_gpu=use_gpu)
    decision = decide_from_anomalies(anomalies, scale_factor=scale_factor, verbose=True)

    print(f"📊 Decision: {'🟥 ЗАРАЖЕНО' if decision['suspicious'] else '🟩 ЧИСТО'}")
    print(f"➕ Triggered: {len(decision['triggered_types'])} types")

    if save_json:
        output = {
            "image_path": image_path,
            "anomalies": anomalies,
            "decision": decision
        }
        json_path = image_path.rsplit(".", 1)[0] + "_analysis.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved: {json_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❗ Использование: python pipeline_test.py path_to_image.jpg")
    else:
        run_pipeline(sys.argv[1])
