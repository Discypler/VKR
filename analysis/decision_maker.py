from analysis import config

def decide_from_anomalies(anomalies, scale_factor=1.0, verbose=False):
    base_thresholds = config.THRESHOLDS
    adjusted_thresholds = {}

    for key, val in base_thresholds.items():
        if scale_factor < 1.0:
            adjusted_thresholds[key] = val * (1.5 - scale_factor)
        elif scale_factor > 2.0:
            adjusted_thresholds[key] = val * 0.8
        else:
            adjusted_thresholds[key] = val

    triggered = []
    summary = {}

    for anomaly in anomalies:
        a_type = anomaly.get("type")
        score = anomaly.get("score", 0)
        summary[a_type] = score
        threshold = adjusted_thresholds.get(a_type)
        if threshold is not None and score > threshold:
            triggered.append((a_type, score))

    suspicious = len(triggered) >= config.MIN_TRIGGERED

    if verbose:
        for t, score in triggered:
            print(f"⚠️ {t} → score={score:.2f} > threshold={adjusted_thresholds[t]:.2f}")

    return {
        "suspicious": suspicious,
        "triggered_types": [t for t, _ in triggered],
        "summary": summary,
        "scale_factor": scale_factor
    }
