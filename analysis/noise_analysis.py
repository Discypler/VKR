# noise_analysis.py

import cv2
import numpy as np
import logging
from utils.format import format_anomaly

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')


def compute_noise_std(image, *args, **kwargs):
    use_gpu = kwargs.get('use_gpu', False)

    if not isinstance(image, np.ndarray):
        image = np.array(image)
    image = image.astype(np.uint8)

    try:
        logger.debug("Вход в compute_noise_std()")

        # Преобразование в grayscale
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image

        if use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            logger.debug("Используется GPU (cv2.cuda) для GaussianBlur и absdiff")

            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(image_gray)

            gpu_blurred = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 0)
            blurred_gpu = gpu_blurred.apply(gpu_img)

            residual_gpu = cv2.cuda.absdiff(gpu_img, blurred_gpu)
            residual = residual_gpu.download()

        else:
            logger.debug("Используется CPU")
            blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
            residual = cv2.absdiff(image_gray, blurred)

        sigma = float(np.std(residual))
        h, w = image_gray.shape

        logger.debug(f"[УСПЕХ] Вычислен шум (std): {sigma:.4f}")
        return [format_anomaly(
            x=w // 2,
            y=h // 2,
            score=sigma,
            anomaly_type="gaussian_noise",
            meta={"std": sigma}
        )]

    except Exception as e:
        logger.error(f"[ОШИБКА] compute_noise_std: {e}")
        raise
