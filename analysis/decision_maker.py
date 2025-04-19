from analysis import config

def compute_meta_score(anomalies, weights):
    score = 0.0
    for anomaly in anomalies:
        a_type = anomaly.get("type")
        val = anomaly.get("score", 0)
        weight = weights.get(a_type, 0)
        score += weight * val
    return score

def decide_from_anomalies(anomalies, scale_factor=1.0, verbose=False):
    base_thresholds = config.THRESHOLDS
    weights = config.META_WEIGHTS
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

    meta_score = compute_meta_score(anomalies, weights)
    suspicious = meta_score > config.META_SCORE_THRESHOLD

    if verbose:
        for t, score in triggered:
            print(f"⚠️ {t} → score={score:.2f} > threshold={adjusted_thresholds[t]:.2f}")
        print(f"🧠 meta_score = {meta_score:.4f} > threshold={config.META_SCORE_THRESHOLD}")

    return {
        "suspicious": suspicious,
        "triggered_types": [t for t, _ in triggered],
        "summary": summary,
        "meta_score": meta_score,
        "scale_factor": scale_factor
    }
