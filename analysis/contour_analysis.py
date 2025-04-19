# contour_analysis.py

import cv2
import numpy as np
import logging
from utils.format import format_anomaly

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

def get_edges(image, *args, **kwargs):
    use_gpu = kwargs.get('use_gpu', False)

    if not isinstance(image, np.ndarray):
        image = np.array(image)
    image = image.astype(np.uint8)

    try:
        logger.debug("[ВХОД] get_edges()")

        # Подготовка RGB
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Каналы
        channels = {
            "r": image_rgb[..., 0],
            "g": image_rgb[..., 1],
            "b": image_rgb[..., 2],
            "gray": cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        }

        results = []

        for cname, cimg in channels.items():
            logger.debug(f"[КАНАЛ] {cname}")

            if use_gpu and hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                logger.debug(f"[GPU] Обработка канала {cname} через CUDA")

                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(cimg)

                # Sobel X
                sobelx_filter = cv2.cuda.createSobelFilter(cv2.CV_8UC1, cv2.CV_32F, 1, 0, ksize=3)
                sobelx_gpu = sobelx_filter.apply(gpu_img)

                # Sobel Y
                sobely_filter = cv2.cuda.createSobelFilter(cv2.CV_8UC1, cv2.CV_32F, 0, 1, ksize=3)
                sobely_gpu = sobely_filter.apply(gpu_img)

                sobelx = sobelx_gpu.download()
                sobely = sobely_gpu.download()
                sobel = np.sqrt(sobelx ** 2 + sobely ** 2)

                # Canny
                canny_detector = cv2.cuda.createCannyEdgeDetector(100, 200)
                canny_gpu = canny_detector.detect(gpu_img)
                canny = canny_gpu.download()
            else:
                logger.debug(f"[CPU] Обработка канала {cname}")
                sobelx = cv2.Sobel(cimg, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(cimg, cv2.CV_64F, 0, 1, ksize=3)
                sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
                canny = cv2.Canny(cimg, 100, 200)

            sobel_score = float(np.mean(sobel))
            canny_score = float(np.mean(canny))

            results.append(format_anomaly(0, 0, sobel_score, "sobel", {"channel": cname}))
            results.append(format_anomaly(0, 0, canny_score, "canny", {"channel": cname}))

        logger.debug("[УСПЕХ] get_edges() завершён успешно")
        return results

    except Exception as e:
        logger.error(f"[ОШИБКА] В get_edges: {e}")
        raise
