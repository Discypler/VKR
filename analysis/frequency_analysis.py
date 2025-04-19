# frequency_analysis.py

import cv2
import numpy as np
import logging

try:
    import cupy as cp
except ImportError:
    cp = None

from utils.format import format_anomaly

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')


def compute_fft_magnitude(image, *args, **kwargs):
    use_gpu = kwargs.get('use_gpu', False)

    if not isinstance(image, np.ndarray):
        image = np.array(image)

    image = image.astype(np.uint8)

    try:
        logger.debug("Вход в compute_fft_magnitude()")

        # Преобразование в RGB и GRAY
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # Выбор библиотеки
        xp = cp if (use_gpu and cp is not None) else np

        def fft_peak_power(channel):
            arr = xp.asarray(channel)
            fft = xp.fft.fft2(arr)
            fft_shift = xp.fft.fftshift(fft)
            magnitude = xp.abs(fft_shift)
            peak = xp.max(magnitude)
            return float(cp.asnumpy(peak) if use_gpu and cp is not None else peak)

        # Вычисляем fft max по каналам
        fft_r = fft_peak_power(image_rgb[..., 0])
        fft_g = fft_peak_power(image_rgb[..., 1])
        fft_b = fft_peak_power(image_rgb[..., 2])
        fft_gray = fft_peak_power(image_gray)

        score_avg = np.mean([fft_r, fft_g, fft_b, fft_gray])
        h, w = image_gray.shape

        logger.debug(f"[УСПЕХ] FFT max values: R={fft_r:.2f}, G={fft_g:.2f}, B={fft_b:.2f}, Gray={fft_gray:.2f}")
        logger.debug("Функция compute_fft_magnitude() выполнена успешно")

        return [format_anomaly(
            x=w // 2,
            y=h // 2,
            score=score_avg,
            anomaly_type="fft_peak",
            meta={
                "fft_r": fft_r,
                "fft_g": fft_g,
                "fft_b": fft_b,
                "fft_gray": fft_gray
            }
        )]

    except Exception as e:
        logger.error(f"Ошибка в compute_fft_magnitude: {e}")
        raise
