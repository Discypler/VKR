
# decision_maker.py

from collections import defaultdict
from analysis import config

def decide_from_anomalies(anomalies, strategy="threshold_sum", verbose=False):
    """
    Простейший решатель, основанный на стратегии объединения аномалий.
    """
    type_scores = defaultdict(list)
    for a in anomalies:
        type_scores[a['type']].append(a['score'])

    # Стратегия: если ≥ 3 типов аномалий превышают порог → заражение
    triggered = []
    for t, scores in type_scores.items():
        mean_score = sum(scores) / len(scores)
        threshold = config.thresholds.get(t, 0)
        if mean_score > threshold:
            triggered.append((t, mean_score))
            if verbose:
                print(f"⚠️ {t} → mean={mean_score:.2f} > threshold={threshold}")

    result = {
        "suspicious": len(triggered) >= 3,
        "triggered_types": triggered,
        "total_types": len(type_scores),
        "summary": {t: sum(v) / len(v) for t, v in type_scores.items()}
    }
    return result
