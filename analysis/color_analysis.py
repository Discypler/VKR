# color_analysis.py

import cv2
import numpy as np
import logging
from utils.format import format_anomaly

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

try:
    import cupy as cp
except ImportError:
    cp = None


def get_saturation_map(image, *args, **kwargs):
    use_gpu = kwargs.get('use_gpu', False)

    if not isinstance(image, np.ndarray):
        image = np.array(image)

    image = image.astype(np.uint8)

    try:
        logger.debug("Вход в get_saturation_map()")

        if use_gpu:
            logger.debug("Используется GPU (cv2.cuda)")
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(image)

            if len(image.shape) == 3:
                gpu_rgb = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2RGB)
                gpu_gray = cv2.cuda.cvtColor(gpu_rgb, cv2.COLOR_RGB2GRAY)
            else:
                gpu_rgb = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_GRAY2RGB)
                gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_GRAY2GRAY)

            image_rgb = gpu_rgb.download()
            image_gray = gpu_gray.download()
        else:
            logger.debug("Используется CPU")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # HSV преобразование
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]

        # Расчёт метрик
        def compute_score(channel):
            mean_val = np.mean(channel)
            std_val = np.std(channel)
            score = std_val / (mean_val + 1e-5)
            return score, mean_val, std_val

        s_score, s_mean, s_std = compute_score(s_channel)
        h_score, h_mean, h_std = compute_score(h_channel)
        v_score, v_mean, v_std = compute_score(v_channel)
        g_score, g_mean, g_std = compute_score(image_gray)

        score_avg = np.mean([s_score, h_score, v_score, g_score])
        h, w = s_channel.shape

        logger.debug("[УСПЕХ] Функция get_saturation_map() выполнена успешно")
        return [format_anomaly(
            w // 2, h // 2, score_avg, "color_saturation", {
                "hue_score": h_score,
                "saturation_score": s_score,
                "value_score": v_score,
                "gray_score": g_score,
                "hue_mean": h_mean,
                "saturation_mean": s_mean,
                "value_mean": v_mean,
                "gray_mean": g_mean,
                "block_size": f"{h}x{w}"
            }
        )]

    except Exception as e:
        logger.error(f"Ошибка в get_saturation_map: {e}")
        raise
