import time
import logging
import numpy as np
import cv2
from skimage import data
from utils.format import format_anomaly

# Анализаторы
from analysis.histogram_analysis import analyze_histograms
from analysis.frequency_analysis import compute_fft_magnitude
from analysis.jpeg_artifact_analysis import detect_jpeg_artifacts
from analysis.contour_analysis import get_edges
from analysis.noise_analysis import compute_noise_std
from analysis.texture_analysis import texture_maps
from analysis.statistical_anomaly import median_residual
from analysis.block_stats import detect_statistical_anomalies
from analysis.combined_detector import decision_by_thresholds
from analysis.pattern_matching import detect_repeating_blocks
from analysis.geometry_analysis import detect_simple_shapes
from analysis.block_similarity import detect_low_correlation_blocks
from analysis.shape_analysis import detect_strange_shapes
from analysis.similarity_analysis import detect_ssim_drop
from analysis.color_analysis import get_saturation_map

# Визуализация
from visualizer.overlay import draw_anomalies

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def run_with_timing(func, *args, name=None, **kwargs):
    logger.info(f"🟡 Тест: {name or func.__name__}")
    start = time.time()
    try:
        result = func(*args, **kwargs)
        logger.info(f"✅ Успешно. Найдено: {len(result)} аномалий. Время: {time.time() - start:.2f} сек.")
        return result
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        return []

def run_tests(use_gpu=False):
    logger.info(f"\n=== 🚀 ПОЛНЫЙ ЗАПУСК АНАЛИЗАТОРОВ ({'GPU' if use_gpu else 'CPU'}) ===")

    img_rgb = data.astronaut()
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    all_anomalies = []

    all_anomalies += run_with_timing(analyze_histograms, img_rgb)
    all_anomalies += run_with_timing(compute_fft_magnitude, img_rgb, use_gpu=use_gpu)
    all_anomalies += run_with_timing(detect_jpeg_artifacts, img_bgr, use_gpu=use_gpu)
    _ = run_with_timing(get_edges, img_rgb, name="get_edges (возвращает sobel, canny)")
    all_anomalies += run_with_timing(compute_noise_std, img_rgb, use_gpu=use_gpu)
    all_anomalies += run_with_timing(detect_statistical_anomalies, img_rgb, use_gpu=use_gpu)
    all_anomalies += run_with_timing(detect_repeating_blocks,img_rgb, use_gpu=use_gpu, block_sizes=[8, 16, 24], max_pairs=100000000, name="detect_repeating_blocks")

    all_anomalies += run_with_timing(detect_simple_shapes, img_rgb)
    all_anomalies += run_with_timing(detect_low_correlation_blocks, img_rgb, use_gpu=use_gpu)
    all_anomalies += run_with_timing(detect_strange_shapes, img_rgb, use_gpu=use_gpu)
    all_anomalies += run_with_timing(detect_ssim_drop, img_rgb, use_gpu=use_gpu)
    all_anomalies += run_with_timing(get_saturation_map, img_rgb, use_gpu=use_gpu)
    all_anomalies += run_with_timing(median_residual, img_rgb, use_gpu=use_gpu)
    all_anomalies += run_with_timing(texture_maps, img_rgb, use_gpu=use_gpu)

    logger.info("\n🧠 Метрики для мета-оценки:")
    metrics = {
        a["type"]: a["score"]
        for a in all_anomalies
        if isinstance(a, dict) and "type" in a and "score" in a
    }
    logger.info(metrics)

    decision = decision_by_thresholds(metrics, {
        "fft_peak": 1000,
        "median_diff_std": 20,
        "block_std": 50,
        "jpeg_artifact": 10
    })

    if decision:
        logger.warning("📌 Итоговое решение: ⚠️ Подозрение на триггер")
    else:
        logger.info("📌 Итоговое решение: ✅ Чисто")

    anomaly_points = [a for a in all_anomalies if isinstance(a, dict) and "x" in a and "y" in a]
    draw_anomalies(img_rgb, anomaly_points, output_path="astronaut_annotated.jpg", show=False)

    logger.info("=== ✅ ЗАВЕРШЕНО ===")

if __name__ == "__main__":
    run_tests(use_gpu=False)
    run_tests(use_gpu=True)
