# pipeline_test.py

import sys
import cv2
import numpy as np
from PIL import Image
from analysis.all_in_one_analyzer import analyze_all
from analysis.decision_maker import decide_from_anomalies

def run_pipeline(image_path, use_gpu=True, return_result=False):
    # Загрузка изображения
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"❌ Невозможно загрузить изображение: {image_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Анализ всех аномалий
    anomalies = analyze_all(img_rgb, use_gpu=use_gpu)

    # Принятие решения
    decision = decide_from_anomalies(anomalies, verbose=True)

    # Вывод
    print(f"📊 Decision: {'🟥 ЗАРАЖЕНО' if decision['suspicious'] else '🟩 ЧИСТО'}")
    print(f"➕ Triggered: {len(decision['triggered_types'])} types")

    result = {
        "image_path": image_path,
        "anomalies": anomalies,
        "decision": decision
    }

    if return_result:
        return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❗ Использование: python pipeline_test.py path_to_image.jpg")
    else:
        run_pipeline(sys.argv[1])
