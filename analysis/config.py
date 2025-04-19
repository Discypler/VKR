
# config.py

# Общие параметры
use_gpu = True  # Использовать GPU, если доступен
block_size = 16  # Размер блока для блочных методов

# Пороговые значения для аномалий
thresholds = {
    "histogram_zero_ratio": 0.15,
    "fft_energy": 1.2e6,
    "jpeg_artifact_score": 5.0,
    "median_diff_std": 5.0,
    "gaussian_noise_std": 4.0,
    "low_correlation_score": 0.4,
    "ssim_drop": 0.25,
    "entropy_mean": 4.0,
    "lbp_var": 40.0,
    "shape_count": 2,
    "strange_shape_count": 1,
    "color_saturation_ratio": 0.3,
}

# Параметры по модулям
parameters = {
    "histogram_analysis": {
        "bins": 256
    },
    "frequency_analysis": {
        "energy_threshold": 1.2e6
    },
    "jpeg_artifact_analysis": {
        "quality_levels": [50, 75, 90]
    },
    "contour_analysis": {
        "edge_threshold": 20
    },
    "noise_analysis": {
        "gaussian_kernel": (7, 7)
    },
    "block_stats": {
        "compare_neighbors": True
    },
    "geometry_analysis": {
        "min_area": 10
    },
    "shape_analysis": {
        "min_area": 10,
        "exclude_shapes": [3, 4, 5]
    },
    "block_similarity": {
        "directions": ["horizontal", "vertical"]
    },
    "similarity_analysis": {
        "ssim_threshold": 0.8
    },
    "color_analysis": {
        "min_saturation": 30
    },
    "statistical_anomaly": {
        "median_kernel": 3
    },
    "texture_analysis": {
        "lbp_radius": 1,
        "lbp_points": 8,
        "entropy_disk": 5
    },
    "noise_sniffer": {
        "block_std_threshold": 4.0
    },
    "ela_analysis": {
        "jpeg_quality": 90,
        "max_diff_threshold": 20
    }
}
