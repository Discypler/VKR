# config.py

THRESHOLDS = {
    "jpeg_artifact": 6.0,
    "ssim_drop": 0.25,
    "median_diff_std": 5.0,
    "noise_sniffer": 5.0,
    "histogram": 1.0,
    "fft_peak": 1e5,
    "color_saturation": 1.0,
    "stat_std_anomaly": 20.0
}

MIN_TRIGGERED = 3

ENABLED_MODULES = [
    "jpeg_artifact",
    "ssim_drop",
    "median_diff_std",
    "noise_sniffer",
    "histogram",
    "fft_peak",
    "color_saturation",
    "stat_std_anomaly",
    "sobel",
    "canny"
]

USE_GPU = True

SAVE_JSON = True
SAVE_CSV = True
SAVE_EVERY = 100
JSON_PATH = "analysis_results.json"
CSV_PATH = "analysis_results.csv"

LOG_LEVEL = "INFO"
