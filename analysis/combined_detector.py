#combined_detector.py

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')


def decision_by_thresholds(metrics_dict, thresholds):
    try:
        logger.debug("[ОТЛАДКА] Вход в decision_by_thresholds()")
        anomaly_votes = 0
        for key in thresholds:
            if key in metrics_dict and metrics_dict[key] > thresholds[key]:
                anomaly_votes += 1

        logger.debug("[УСПЕХ] Функция gdecision_by_thresholds() выполнена успешно")
        return anomaly_votes >= 2

    except Exception as e:
        logger.error(f"Ошибка в get_saturation_map: {e}")
        raise
