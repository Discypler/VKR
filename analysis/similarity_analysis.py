import cv2
import numpy as np
import logging
from skimage.metrics import structural_similarity as ssim
from utils.format import format_anomaly

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

def detect_ssim_drop(image, *args, **kwargs):
    use_gpu = kwargs.get('use_gpu', False)

    if not isinstance(image, np.ndarray):
        image = np.array(image)
    image = image.astype(np.uint8)

    try:
        logger.debug("Вход в detect_ssim_drop()")

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
            if use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                logger.debug(f"Используется GPU для канала {cname}")
                gpu_mat = cv2.cuda_GpuMat()
                gpu_mat.upload(cimg)

                blur = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (9, 9), 0)
                blurred_gpu = blur.apply(gpu_mat)
                blurred = blurred_gpu.download()
            else:
                blurred = cv2.GaussianBlur(cimg, (9, 9), 0)

            ssim_val = ssim(cimg, blurred)
            h, w = cimg.shape
            anomalies.append(format_anomaly(
                x=w // 2,
                y=h // 2,
                score=1.0 - ssim_val,
                anomaly_type="ssim_drop",
                meta={
                    "ssim": ssim_val,
                    "channel": cname
                }
            ))

        logger.debug(f"[УСПЕХ] Найдено {len(anomalies)} SSIM метрик")
        return anomalies

    except Exception as e:
        logger.error(f"[ОШИБКА] detect_ssim_drop: {e}")
        raise
