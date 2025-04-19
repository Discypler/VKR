# config.py

THRESHOLDS = {
    "jpeg_artifact": 6.0,
    "ssim_drop": 0.25,
    "median_diff_std": 5.0,
    "noise_sniffer": 6.0,
    "histogram": 1.0,
    "fft_peak": 1.5e5,
    "color_saturation": 1.0,
    "stat_std_anomaly": 20.0,
    "sobel": 125.0,
    "canny": 45.0,
    "ela_score": 12.0,
    "texture_entropy_mean": 4.5,
    "lbp_var": 8.0,
    "shape_count": 2,
    "strange_shape_count": 1,
    "low_correlation_score": 0.3
}

MIN_TRIGGERED = 3

ENABLED_MODULES = list(THRESHOLDS.keys())

USE_GPU = True

SAVE_JSON = True
SAVE_CSV = True
SAVE_EVERY = 100
JSON_PATH = "analysis_results.json"
CSV_PATH = "analysis_results.csv"

LOG_LEVEL = "INFO"

parameters = {
    "block_size": 16,
    "ela_quality": 90,
    "ela_difference_threshold": 8
}
