
import cv2
import numpy as np

def draw_anomalies(image, anomalies, output_path=None, show=True):
    try:
        print("[ОТЛАДКА] Вход в draw_anomalies()")
        image_copy = image.copy()
        for anomaly in anomalies:
            if not isinstance(anomaly, dict):
                continue
            x, y = anomaly['x'], anomaly['y']
            score = anomaly.get('score', 0)
            cv2.circle(image_copy, (int(x), int(y)), 4, (0, 0, 255), -1)
            cv2.putText(image_copy, f"{score:.1f}", (int(x)+5, int(y)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        if output_path:
            cv2.imwrite(output_path, image_copy)
        if show:
            cv2.imshow("Anomalies", image_copy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception as e:
        print("[ОШИБКА] В draw_anomalies:", e)
        raise
