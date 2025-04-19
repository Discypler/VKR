# jpeg_artifact_analysis.py

import cv2
import numpy as np
import logging
from utils.format import format_anomaly

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')


def detect_jpeg_artifacts(image, *args, **kwargs):
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    image = image.astype(np.uint8)

    try:
        logger.debug("Вход в detect_jpeg_artifacts()")

        # Преобразование в BGR (OpenCV требует BGR для imencode)
        if len(image.shape) == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), cv2.COLOR_RGB2BGR)

        # Кодирование и декодирование JPEG (качество ↓)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        result, encimg = cv2.imencode('.jpg', image_bgr, encode_param)
        decimg = cv2.imdecode(encimg, 1)

        # Подсчёт остаточного шума между оригиналом и JPEG
        residual = cv2.absdiff(image_bgr, decimg)
        artifact_score = float(np.mean(residual))
        h, w = residual.shape[:2]

        logger.debug(f"[УСПЕХ] JPEG artifact score = {artifact_score:.2f}")
        return [format_anomaly(
            x=w // 2,
            y=h // 2,
            score=artifact_score,
            anomaly_type="jpeg_artifact",
            meta={"jpeg_quality": 50}
        )]

    except Exception as e:
        logger.error(f"[ОШИБКА] detect_jpeg_artifacts: {e}")
        raise