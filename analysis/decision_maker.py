def decide_from_anomalies(anomalies, scale_factor=1.0, verbose=False):
    base_thresholds = {
        "jpeg_artifact": 6.0,
        "ssim_drop": 0.25,
        "median_diff_std": 5.0,
        "noise_sniffer": 5.0,
        "histogram": 1.0,
        "fft_peak": 1e5,
        "color_saturation": 1.0,
        "stat_std_anomaly": 20.0
    }

    adjusted_thresholds = {}
    for key, val in base_thresholds.items():
        if scale_factor < 1.0:
            adjusted_thresholds[key] = val * (1.5 - scale_factor)
        elif scale_factor > 2.0:
            adjusted_thresholds[key] = val * 0.8
        else:
            adjusted_thresholds[key] = val

    triggered = []
    for anomaly in anomalies:
        a_type = anomaly.get("type")
        score = anomaly.get("score", 0)
        threshold = adjusted_thresholds.get(a_type)
        if threshold is not None and score > threshold:
            triggered.append((a_type, score))

    suspicious = len(triggered) >= 3

    if verbose:
        for t, score in triggered:
            print(f"⚠️ {t} → score={score:.2f} > threshold={adjusted_thresholds[t]:.2f}")

    return {
        "suspicious": suspicious,
        "triggered_types": [t for t, _ in triggered]
    }
