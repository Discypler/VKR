# block_stats.py

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


def detect_statistical_anomalies(image, *args, **kwargs):
    std_thresh = kwargs.get('std_thresh', 25.0)
    use_gpu = kwargs.get('use_gpu', False)
    block_size = kwargs.get('block_size', 16)

    if not isinstance(image, np.ndarray):
        image = np.array(image)

    try:
        logger.debug(f"Вход в detect_statistical_anomalies() с block_size = {block_size}")
        anomalies = []

        # Преобразуем изображение (cv2.cuda или обычный)
        if use_gpu and cp is not None and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            logger.debug("Используется GPU: cv2.cuda + cupy")
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
            xp = cp
        else:
            logger.debug("Используется CPU")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            xp = np

        h, w = image_gray.shape

        # Вычисление std по блокам
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                b_r = image_rgb[y:y + block_size, x:x + block_size, 0]
                b_g = image_rgb[y:y + block_size, x:x + block_size, 1]
                b_b = image_rgb[y:y + block_size, x:x + block_size, 2]
                b_gray = image_gray[y:y + block_size, x:x + block_size]

                std_r = xp.std(xp.asarray(b_r))
                std_g = xp.std(xp.asarray(b_g))
                std_b = xp.std(xp.asarray(b_b))
                std_gray = xp.std(xp.asarray(b_gray))

                if use_gpu and cp is not None:
                    std_r = float(cp.asnumpy(std_r))
                    std_g = float(cp.asnumpy(std_g))
                    std_b = float(cp.asnumpy(std_b))
                    std_gray = float(cp.asnumpy(std_gray))

                std_avg = np.mean([std_r, std_g, std_b, std_gray])

                if std_avg > std_thresh:
                    anomalies.append(format_anomaly(
                        x + block_size // 2,
                        y + block_size // 2,
                        std_avg,
                        "stat_std_anomaly",
                        {
                            "std_r": std_r,
                            "std_g": std_g,
                            "std_b": std_b,
                            "std_gray": std_gray,
                            "block_size": block_size
                        }
                    ))

        logger.debug(f"Найдено аномалий: {len(anomalies)}")
        logger.debug("[УСПЕХ] Функция detect_statistical_anomalies() выполнена успешно")
        return anomalies

    except Exception as e:
        logger.error(f"Ошибка в detect_statistical_anomalies: {e}")
        raise
