# all_in_one_analyzer.py

import importlib
from analysis import config
from utils.format import format_anomaly
from concurrent.futures import ThreadPoolExecutor
from analysis.full_feature_extractor import prepare_image


# Модули анализа
MODULES = {
    "histogram": "analysis.histogram_analysis.analyze_histograms",
    "fft": "analysis.frequency_analysis.compute_fft_magnitude",
    "jpeg": "analysis.jpeg_artifact_analysis.detect_jpeg_artifacts",
    "contour": "analysis.contour_analysis.get_edges",
    "noise": "analysis.noise_analysis.compute_noise_std",
    "block_stats": "analysis.block_stats.detect_statistical_anomalies",
    "geometry": "analysis.geometry_analysis.detect_simple_shapes",
    "shape": "analysis.shape_analysis.detect_strange_shapes",
    "block_similarity": "analysis.block_similarity.detect_low_correlation_blocks",
    "ssim": "analysis.similarity_analysis.detect_ssim_drop",
    "color": "analysis.color_analysis.get_saturation_map",
    "stat": "analysis.statistical_anomaly.median_residual",
    "texture": "analysis.texture_analysis.texture_maps",
    "noise_sniffer": "analysis.noise_sniffer.analyze_residual_noise",
    "ela": "analysis.ela_analysis.analyze_ela"
}


def dynamic_import(func_path):
    mod, func = func_path.rsplit(".", 1)
    module = importlib.import_module(mod)
    return getattr(module, func)

def run_module(func, image_rgb, image_bgr, image_gray, use_gpu):
    try:
        if func.__code__.co_argcount == 2:
            return func(image_rgb, use_gpu)
        else:
            return func(image_rgb, image_bgr, image_gray, use_gpu)
    except Exception as e:
        print(f"[ERROR] {func.__name__}: {e}")
        return []

def analyze_all(image, use_gpu=True):
    image_rgb, image_bgr, image_gray = prepare_image(image)
    results = []

    with ThreadPoolExecutor() as executor:
        futures = {}
        for name, path in MODULES.items():
            func = dynamic_import(path)
            futures[executor.submit(run_module, func, image_rgb, image_bgr, image_gray, use_gpu)] = name

        for future in futures:
            try:
                output = future.result()
                if isinstance(output, list):
                    results.extend(output)
            except Exception as ex:
                print(f"[ERROR] Exception in {futures[future]}: {ex}")
    return results
