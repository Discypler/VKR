# shape_analysis.py

import cv2
import numpy as np
import logging
from utils.format import format_anomaly

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')


def detect_strange_shapes(image, *args, **kwargs):
    use_gpu = kwargs.get('use_gpu', False)

    if not isinstance(image, np.ndarray):
        image = np.array(image)
    image = image.astype(np.uint8)

    anomalies = []

    try:
        logger.debug("Вход в detect_strange_shapes()")

        # Преобразование в grayscale
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image

        if use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            logger.debug("Используется GPU для GaussianBlur и Threshold")
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(image_gray)

            # Размытие
            blur = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (3, 3), 0)
            blurred_gpu = blur.apply(gpu_mat)

            # Порог
            _, thresh_gpu = cv2.cuda.threshold(blurred_gpu, 127, 255, cv2.THRESH_BINARY)
            thresh = thresh_gpu.download()
        else:
            logger.debug("Используется CPU")
            image_gray = cv2.GaussianBlur(image_gray, (3, 3), 0)
            _, thresh = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        strange_shapes = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10:
                continue
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            if len(approx) not in [3, 4, 5]:
                strange_shapes += 1

        if strange_shapes > 0:
            anomalies.append(format_anomaly(
                x=0,
                y=0,
                score=strange_shapes,
                anomaly_type="strange_shape",
                meta={"count": strange_shapes}
            ))

        logger.debug(f"[УСПЕХ] Найдено странных форм: {strange_shapes}")
        return anomalies

    except Exception as e:
        logger.error(f"[ОШИБКА] detect_strange_shapes: {e}")
        raise
