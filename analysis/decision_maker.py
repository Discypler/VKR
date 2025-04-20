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
    weights = config.META_WEIGHTS
    summary = {}

    for anomaly in anomalies:
        a_type = anomaly.get("type")
        summary[a_type] = anomaly.get("score", 0)

    meta_score = compute_meta_score(anomalies, weights)
    suspicious = meta_score > config.META_SCORE_THRESHOLD

    if verbose:
        print(f"🧠 meta_score = {meta_score:.4f} > threshold={config.META_SCORE_THRESHOLD}")

    return {
        "suspicious": suspicious,
        "triggered_types": [a["type"] for a in anomalies if a["type"] in weights],
        "summary": summary,
        "meta_score": meta_score,
        "scale_factor": scale_factor
    }
