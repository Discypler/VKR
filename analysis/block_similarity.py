#block_similarity.py

import cv2
import numpy as np
import logging
from utils.format import format_anomaly

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')


def detect_low_correlation_blocks(image, *args, **kwargs):
    block_size = kwargs.get('block_size', 16)
    use_gpu = kwargs.get('use_gpu', False)

    if not isinstance(image, np.ndarray):
        image = np.array(image)

    anomalies = []

    try:
        logger.debug("Вход в detect_low_correlation_blocks()")

        if use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            logger.debug("Используется GPU через cv2.cuda")

            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(image)

            # Преобразование в RGB и Grayscale
            if len(image.shape) == 3:
                gpu_rgb = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2RGB)
                gpu_gray = cv2.cuda.cvtColor(gpu_rgb, cv2.COLOR_RGB2GRAY)
            else:
                gpu_rgb = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_GRAY2RGB)
                gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_GRAY2GRAY)

            # Скачиваем обратно на CPU для анализа (т.к. np.corrcoef не работает на GPU)
            image_rgb = gpu_rgb.download()
            image_gray = gpu_gray.download()
        else:
            logger.debug("Используется CPU")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        h, w, _ = image_rgb.shape

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - 2 * block_size, block_size):
                # Блоки по каждому каналу
                b1 = image_rgb[y:y+block_size, x:x+block_size]
                b2 = image_rgb[y:y+block_size, x+block_size:x+2*block_size]

                # Корреляция по каждому каналу
                corr_r = np.corrcoef(b1[..., 0].flatten(), b2[..., 0].flatten())[0, 1]
                corr_g = np.corrcoef(b1[..., 1].flatten(), b2[..., 1].flatten())[0, 1]
                corr_b = np.corrcoef(b1[..., 2].flatten(), b2[..., 2].flatten())[0, 1]

                # Дополнительно — grayscale
                b1_gray = image_gray[y:y+block_size, x:x+block_size]
                b2_gray = image_gray[y:y+block_size, x+block_size:x+2*block_size]
                corr_gray = np.corrcoef(b1_gray.flatten(), b2_gray.flatten())[0, 1]

                # Среднее значение корреляции
                corr_avg = np.mean([corr_r, corr_g, corr_b, corr_gray])

                if corr_avg < 0.5:
                    anomalies.append(format_anomaly(
                        x + block_size // 2,
                        y + block_size // 2,
                        corr_avg,
                        "low_corr",
                        {
                            "corr_r": float(corr_r),
                            "corr_g": float(corr_g),
                            "corr_b": float(corr_b),
                            "corr_gray": float(corr_gray),
                            "block_size": block_size
                        }
                    ))

        logger.debug(f"Найдено аномалий: {len(anomalies)}")
        logger.debug("[УСПЕХ] Функция detect_low_correlation_blocks() выполнена успешно")
        return anomalies

    except Exception as e:
        logger.error(f"Ошибка в detect_low_correlation_blocks: {e}")
        raise
