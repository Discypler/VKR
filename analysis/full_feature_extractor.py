import numpy as np
import cv2
import logging

# ✅ Функция для подготовки изображений
def prepare_image(image):
    if isinstance(image, np.ndarray):
        image_rgb = image
    else:
        from PIL import Image
        image_rgb = np.array(image.convert("RGB"))

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return image_rgb, image_bgr, image_gray

# Импорты модулей анализа
from analysis.histogram_analysis import analyze_histograms
from analysis.frequency_analysis import compute_fft_magnitude
from analysis.jpeg_artifact_analysis import detect_jpeg_artifacts
from analysis.contour_analysis import get_edges
from analysis.noise_analysis import compute_noise_std
from analysis.block_stats import detect_statistical_anomalies
from analysis.geometry_analysis import detect_simple_shapes
from analysis.block_similarity import detect_low_correlation_blocks
from analysis.shape_analysis import detect_strange_shapes
from analysis.similarity_analysis import detect_ssim_drop
from analysis.color_analysis import get_saturation_map
from analysis.statistical_anomaly import median_residual
from analysis.texture_analysis import texture_maps

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

def extract_all_features(image, use_gpu=False):
    logger.debug("Вход в extract_all_features()")

    if not isinstance(image, np.ndarray):
        image = np.array(image)
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    image_rgb = image.copy()
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

    flat_results = {}
    detailed_results = {}

    analyzers = [
        ("histogram", analyze_histograms),
        ("fft_peak", compute_fft_magnitude),
        ("jpeg_artifact", detect_jpeg_artifacts),
        ("sobel_canny", get_edges),
        ("gaussian_noise", compute_noise_std),
        ("stat_std_anomaly", detect_statistical_anomalies),
        ("simple_shape", detect_simple_shapes),
        ("low_corr", detect_low_correlation_blocks),
        ("strange_shape", detect_strange_shapes),
        ("ssim_drop", detect_ssim_drop),
        ("color_saturation", get_saturation_map),
        ("median_diff_std", median_residual),
        ("texture_analysis", texture_maps)
    ]

    for name, func in analyzers:
        try:
            result = func(image, use_gpu=use_gpu)

            if isinstance(result, list):
                valid = [r for r in result if isinstance(r, dict) and "type" in r and "score" in r]
                if valid:
                    anomaly_type = valid[0]["type"]
                    flat_results[anomaly_type] = float(np.mean([v["score"] for v in valid]))
                    detailed_results[anomaly_type] = valid
            elif isinstance(result, dict):
                anomaly_type = result.get("type", name)
                flat_results[anomaly_type] = result.get("score", 0)
                detailed_results[anomaly_type] = [result]
            else:
                logger.warning(f"{name} вернул неожиданный формат: {type(result)}")
        except Exception as e:
            logger.error(f"[ОШИБКА] В {name}: {e}")

    return flat_results, detailed_results
