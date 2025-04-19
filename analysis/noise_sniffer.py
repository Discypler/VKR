
# noise_sniffer.py

import numpy as np
import cv2
from analysis import config
from utils.format import format_anomaly

def analyze_residual_noise(image_rgb, use_gpu=False):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    if use_gpu:
        gray_gpu = cv2.cuda_GpuMat()
        gray_gpu.upload(gray)
        gauss_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (7, 7), 1.5)
        blurred_gpu = gauss_filter.apply(gray_gpu)
        blurred = blurred_gpu.download()
    else:
        blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

    residual = cv2.absdiff(gray, blurred)

    # Разделение на блоки
    h, w = residual.shape
    bs = config.block_size
    anomalies = []
    for y in range(0, h, bs):
        for x in range(0, w, bs):
            block = residual[y:y+bs, x:x+bs]
            if block.size == 0:
                continue
            std_val = np.std(block)
            if std_val > config.thresholds["gaussian_noise_std"]:
                anomalies.append(format_anomaly(x, y, std_val, "noise_sniffer", {
                    "std": float(std_val)
                }))
    return anomalies
