#texture_analysis.py

import cv2
import numpy as np
import logging
from skimage.feature import local_binary_pattern
from skimage.filters.rank import entropy
from skimage.morphology import disk
from utils.format import format_anomaly

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')


def texture_maps(image, *args, **kwargs):
    use_gpu = kwargs.get('use_gpu', False)

    if not isinstance(image, np.ndarray):
        image = np.array(image)
    image = image.astype(np.uint8)

    try:
        logger.debug("Вход в texture_maps()")

        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image

        channels = {
            "r": image_rgb[..., 0],
            "g": image_rgb[..., 1],
            "b": image_rgb[..., 2],
            "gray": cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        }

        anomalies = []

        for cname, cimg in channels.items():
            logger.debug(f"Анализ текстур для канала {cname}")
            lbp = local_binary_pattern(cimg, P=8, R=1, method='uniform')
            lbp_var = float(np.var(lbp))

            ent = entropy(cimg, disk(5))
            ent_mean = float(np.mean(ent))

            h, w = cimg.shape
            anomalies.append(format_anomaly(
                x=w // 2,
                y=h // 2,
                score=lbp_var,
                anomaly_type="texture_lbp_var",
                meta={"channel": cname, "var": lbp_var}
            ))
            anomalies.append(format_anomaly(
                x=w // 2,
                y=h // 2,
                score=ent_mean,
                anomaly_type="texture_entropy_mean",
                meta={"channel": cname, "mean": ent_mean}
            ))

        logger.debug(f"[УСПЕХ] texture_maps завершён. Всего метрик: {len(anomalies)}")
        return anomalies

    except Exception as e:
        logger.error(f"[ОШИБКА] texture_maps: {e}")
        raise
