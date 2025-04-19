# histogram_analysis.py

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


def analyze_histograms(image, *args, **kwargs):
    use_gpu = kwargs.get('use_gpu', False)
    anomalies = []

    if not isinstance(image, np.ndarray):
        image = np.array(image)

    image = image.astype(np.uint8)

    try:
        logger.debug("Вход в analyze_histograms()")

        # Преобразования
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        xp = cp if (use_gpu and cp is not None) else np

        def extract_histogram(channel, name):
            arr = xp.asarray(channel)
            hist = xp.histogram(arr, bins=256, range=(0, 256))[0]
            if use_gpu and cp is not None:
                hist = cp.asnumpy(hist)
            return hist

        def analyze_hist(hist, name):
            peak = int(np.max(hist))
            spread = int(np.sum(hist > 0))
            zeros = int(np.sum(hist == 0))
            return format_anomaly(
                x=0, y=0, score=zeros,
                anomaly_type="histogram",
                meta={
                    "channel": name,
                    "zeros": zeros,
                    "spread": spread,
                    "peak": peak
                }
            )

        # По каналам
        for idx, name in enumerate(["r", "g", "b"]):
            hist = extract_histogram(image_rgb[:, :, idx], name)
            anomalies.append(analyze_hist(hist, name))

        # Gray
        hist_gray = extract_histogram(image_gray, "gray")
        anomalies.append(analyze_hist(hist_gray, "gray"))

        logger.debug("[УСПЕХ] analyze_histograms выполнен успешно")
        return anomalies

    except Exception as e:
        logger.error(f"[ОШИБКА] analyze_histograms: {e}")
        raise
