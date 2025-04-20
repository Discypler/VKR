# config.py

THRESHOLDS = {
    "gaussian_noise": 6.0,
    "color_saturation": 1.0,
    "sobel": 125.0,
    "median_diff_std": 5.0,
    "texture_entropy_mean": 4.5
}

ENABLED_MODULES = list(THRESHOLDS.keys())
USE_GPU = True
SAVE_JSON = True
SAVE_CSV = True
SAVE_EVERY = 100
JSON_PATH = "analysis_results.json"
CSV_PATH = "analysis_results.csv"
LOG_LEVEL = "INFO"

META_WEIGHTS = {
    "gaussian_noise": 0.0907,
    "color_saturation": 0.0895,
    "sobel": 0.0825,
    "median_diff_std": 0.0808,
    "texture_entropy_mean": 0.0806
}

META_SCORE_THRESHOLD = 5.0
