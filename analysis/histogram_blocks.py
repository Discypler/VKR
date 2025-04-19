# histogram_blocks.py

import cv2
import numpy as np
import logging
from utils.format import format_anomaly

try:
    import cupy as cp
except ImportError:
    cp = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')


def analyze_histogram_blocks(image, *args, **kwargs):
    use_gpu = kwargs.get('use_gpu', False)
    block_size = kwargs.get('block_size', 32)
    threshold = kwargs.get('threshold', 2.0)

    if not isinstance(image, np.ndarray):
        image = np.array(image)
    image = image.astype(np.uint8)

    try:
        logger.debug("Вход в analyze_histogram_blocks()")

        if use_gpu and cp is not None and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            logger.debug("Используется GPU (cv2.cuda + cupy)")
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(image)
            if len(image.shape) == 3:
                gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_RGB2GRAY)
            else:
                gpu_gray = gpu_img
            image_gray = gpu_gray.download()
            xp = cp
        else:
            logger.debug("Используется CPU")
            if len(image.shape) == 3:
                image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                image_gray = image
            xp = np

        h, w = image_gray.shape
        anomalies = []

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = image_gray[y:y + block_size, x:x + block_size]
                hist = xp.histogram(xp.asarray(block), bins=256, range=(0, 256))[0]
                if use_gpu and cp is not None:
                    hist = cp.asnumpy(hist)

                zeros = np.sum(hist == 0)
                spread = np.sum(hist > 0)
                score = zeros / (spread + 1e-5)

                if score > threshold:
                    anomalies.append(format_anomaly(
                        x + block_size // 2,
                        y + block_size // 2,
                        score,
                        "histogram_block",
                        {
                            "zeros": int(zeros),
                            "spread": int(spread),
                            "block_size": block_size
                        }
                    ))

        logger.debug(f"[УСПЕХ] Найдено {len(anomalies)} подозрительных блоков")
        return anomalies

    except Exception as e:
        logger.error(f"[ОШИБКА] analyze_histogram_blocks: {e}")
        raise
