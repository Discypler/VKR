
# ela_analysis.py

import numpy as np
import cv2
from PIL import Image, ImageChops
import io
from analysis import config
from utils.format import format_anomaly

def analyze_ela(image_rgb, use_gpu=False):
    pil_image = Image.fromarray(image_rgb)
    
    # Сохраняем в буфер в JPEG с заданным качеством
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=config.parameters['ela_analysis']['jpeg_quality'])
    buffer.seek(0)
    recompressed = Image.open(buffer)

    # Вычисляем разницу
    ela_image = ImageChops.difference(pil_image, recompressed)
    ela_np = np.array(ela_image.convert("L"))

    max_diff = np.max(ela_np)
    mean_diff = np.mean(ela_np)

    h, w = ela_np.shape
    block_size = config.block_size
    anomalies = []
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = ela_np[y:y+block_size, x:x+block_size]
            if block.size == 0:
                continue
            max_b = np.max(block)
            if max_b > config.parameters['ela_analysis']['max_diff_threshold']:
                anomalies.append(format_anomaly(x, y, max_b, "ela_max_diff", {
                    "mean": float(np.mean(block)),
                    "max": float(max_b)
                }))
    return anomalies
