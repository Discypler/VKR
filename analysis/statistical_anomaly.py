import cv2
import numpy as np
import logging
from utils.format import format_anomaly

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')


def median_residual(image, *args, **kwargs):
    use_gpu = kwargs.get('use_gpu', False)

    if not isinstance(image, np.ndarray):
        image = np.array(image)
    image = image.astype(np.uint8)

    try:
        logger.debug("Вход в median_residual()")

        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image

        # Каналы для анализа
        channels = {
            "r": image_rgb[..., 0],
            "g": image_rgb[..., 1],
            "b": image_rgb[..., 2],
            "gray": cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        }

        anomalies = []

        for cname, cimg in channels.items():
            if use_gpu and hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                logger.debug(f"[GPU] Применение медианного фильтра для канала {cname}")
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(cimg)

                median_filter = cv2.cuda.createMedianFilter(cv2.CV_8UC1, 3)
                blurred_gpu = median_filter.apply(gpu_img)
                median = blurred_gpu.download()
            else:
                logger.debug(f"[CPU] Применение медианного фильтра для канала {cname}")
                median = cv2.medianBlur(cimg, 3)

            diff = cv2.absdiff(cimg, median)
            std = float(np.std(diff))
            h, w = cimg.shape

            anomalies.append(format_anomaly(
                x=w // 2,
                y=h // 2,
                score=std,
                anomaly_type="median_diff_std",
                meta={
                    "channel": cname,
                    "std": std
                }
            ))

        logger.debug(f"[УСПЕХ] Выполнено median_residual. Всего аномалий: {len(anomalies)}")
        return anomalies

    except Exception as e:
        logger.error(f"[ОШИБКА] В median_residual: {e}")
        raise
