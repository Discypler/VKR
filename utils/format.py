
def format_anomaly(x, y, score, anomaly_type, meta=None):
    if meta is None:
        meta = {}
    return {
        "x": int(x),
        "y": int(y),
        "score": float(score),
        "type": str(anomaly_type),
        "meta": meta
    }
