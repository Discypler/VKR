import cv2
import numpy as np
import logging
from utils.format import format_anomaly

try:
    import cupy as cp
except ImportError:
    cp = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')


def detect_repeating_blocks(image, *args, **kwargs):
    use_gpu = kwargs.get('use_gpu', False)
    max_pairs = kwargs.get('max_pairs', 1000000)
    threshold = kwargs.get('threshold', 0.99)
    block_sizes = kwargs.get('block_sizes', [8, 16, 24])

    if not isinstance(image, np.ndarray):
        image = np.array(image)
    image = image.astype(np.uint8)

    try:
        logger.debug("Вход в detect_repeating_blocks()")

        # Подготовка каналов
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

        xp = cp if use_gpu and cp is not None else np
        all_matches = []

        total_combinations = len(block_sizes) * len(channels)
        max_pairs_per_case = max(1, max_pairs // total_combinations)

        for block_size in block_sizes:
            logger.debug(f"▶ Анализ блоков {block_size}x{block_size}")

            for cname, cimg in channels.items():
                h, w = cimg.shape
                blocks, coords = [], []

                for y in range(0, h - block_size, block_size):
                    for x in range(0, w - block_size, block_size):
                        block = cimg[y:y + block_size, x:x + block_size].flatten()
                        blocks.append(block)
                        coords.append((x, y))

                blocks = xp.asarray(blocks)
                checked = 0

                for i in range(len(blocks)):
                    for j in range(i + 1, len(blocks)):
                        if checked >= max_pairs_per_case:
                            logger.debug(f"[ПРЕРЫВ] {cname}, {block_size} — лимит {max_pairs_per_case} достигнут")
                            break

                        b1 = blocks[i]
                        b2 = blocks[j]

                        if xp.all(b1 == b1[0]) or xp.all(b2 == b2[0]):
                            continue

                        dot = xp.dot(b1, b2)
                        norm = xp.linalg.norm(b1) * xp.linalg.norm(b2)
                        if norm == 0:
                            continue

                        corr = dot / norm
                        if use_gpu and cp is not None:
                            corr = float(cp.asnumpy(corr))

                        if corr > threshold:
                            x, y = coords[i]
                            mx, my = coords[j]
                            all_matches.append(format_anomaly(
                                x=x,
                                y=y,
                                score=corr,
                                anomaly_type="repeating_block",
                                meta={
                                    "match": (mx, my),
                                    "channel": cname,
                                    "block_size": block_size
                                }
                            ))
                        checked += 1

                logger.debug(f"🔍 {cname} / {block_size}: проверено {checked} пар")

        logger.debug(f"[УСПЕХ] Всего совпадений найдено: {len(all_matches)}")
        return all_matches

    except Exception as e:
        logger.error(f"[ОШИБКА] detect_repeating_blocks: {e}")
        raise
