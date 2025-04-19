# geometry_analysis.py

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

def detect_simple_shapes(image, *args, **kwargs):
    use_gpu = kwargs.get('use_gpu', False)

    if not isinstance(image, np.ndarray):
        image = np.array(image)
    image = image.astype(np.uint8)

    try:
        logger.debug("Вход в detect_simple_shapes()")

        if use_gpu and hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            logger.debug("Используется GPU: cv2.cuda")

            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(image)

            if len(image.shape) == 3:
                gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_RGB2GRAY)
            else:
                gpu_gray = gpu_img

            gauss = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 1.5)
            gpu_blurred = gauss.apply(gpu_gray)
            image_gray = gpu_blurred.download()

        else:
            logger.debug("Используется CPU")
            if len(image.shape) == 3:
                image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                image_gray = image
            image_gray = cv2.GaussianBlur(image_gray, (5, 5), 1.5)

        # Порог и контуры — на CPU
        _, binary = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        count = 0
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            if len(approx) in [3, 4, 5]:  # треугольники, прямоугольники, пятиугольники
                count += 1

        logger.debug(f"[УСПЕХ] Найдено простых фигур: {count}")
        return [{
            "type": "simple_shape",
            "score": count,
            "meta": {
                "shapes": count
            }
        }]

    except Exception as e:
        logger.error(f"[ОШИБКА] detect_simple_shapes: {e}")
        return []
