import os
from analysis.pipeline_test import run_pipeline

# 📁 Укажи здесь свою папку с изображениями
IMAGE_DIR = os.path.join("data", "poisoned", "cifar10")

# ⚙️ Расширения, которые будем обрабатывать
VALID_EXTENSIONS = [".jpg", ".jpeg", ".png"]

def collect_images(folder):
    images = []
    for root, _, files in os.walk(folder):
        for f in files:
            if any(f.lower().endswith(ext) for ext in VALID_EXTENSIONS):
                images.append(os.path.join(root, f))
    return images

def main():
    all_images = collect_images(IMAGE_DIR)
    print(f"🔍 Найдено изображений: {len(all_images)}\n")

    for i, path in enumerate(all_images):
        print(f"🔄 [{i+1}/{len(all_images)}] Обработка: {path}")
        try:
            run_pipeline(path, use_gpu=True)
        except Exception as e:
            print(f"❌ Ошибка при обработке {path}: {e}")

if __name__ == "__main__":
    main()
