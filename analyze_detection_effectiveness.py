import os
import json
import csv
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

TRIGGER_SIZE = 8
CHUNK_SIZE = 100
INPUT_JSON = "data/poisoned/features_cifar10_all_attacks.json"
OUTPUT_CSV = "data/poisoned/evaluation_results.csv"

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def count_existing_rows(csv_path):
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f) - 1

def write_to_csv(path, rows, fieldnames, append=False):
    mode = 'a' if append and os.path.exists(path) else 'w'
    write_header = not append or not os.path.exists(path)
    with open(path, mode, encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

def analyze(data, output_csv):
    logger.info("🔍 Старт анализа JSON результатов...")

    already_saved = count_existing_rows(output_csv)
    logger.info(f"🔢 Уже записано строк в CSV: {already_saved}")

    fields = [
        "dataset", "class", "index", "attack_type", "trigger_position",
        "trigger_x", "trigger_y", "total_anomalies", "hits_in_trigger", "hit_rate"
    ]

    buffer = []
    saved = already_saved

    for i, item in enumerate(data):
        if i < already_saved:
            continue

        if item.get("type") != "poisoned":
            continue

        trigger_x = item.get("trigger_x", -1)
        trigger_y = item.get("trigger_y", -1)
        anomalies_by_metric = item.get("anomalies", {})

        total = 0
        hits = 0

        if isinstance(anomalies_by_metric, dict):
            for metric, anomaly_list in anomalies_by_metric.items():
                if isinstance(anomaly_list, list):
                    for a in anomaly_list:
                        if isinstance(a, dict):
                            x = a.get("x")
                            y = a.get("y")
                            if isinstance(x, int) and isinstance(y, int):
                                total += 1
                                if (trigger_x <= x <= trigger_x + TRIGGER_SIZE and
                                    trigger_y <= y <= trigger_y + TRIGGER_SIZE):
                                    hits += 1

        hit_rate = round(hits / total, 4) if total > 0 else 0.0

        row = {
            "dataset": item.get("dataset", ""),
            "class": item.get("class", ""),
            "index": item.get("index", -1),
            "attack_type": item.get("attack_type", ""),
            "trigger_position": item.get("trigger_position", ""),
            "trigger_x": trigger_x,
            "trigger_y": trigger_y,
            "total_anomalies": total,
            "hits_in_trigger": hits,
            "hit_rate": hit_rate
        }

        buffer.append(row)
        saved += 1

        if saved % CHUNK_SIZE == 0:
            write_to_csv(output_csv, buffer, fields, append=True)
            logger.info(f"💾 Сохранено {saved} строк")
            buffer = []

    if buffer:
        write_to_csv(output_csv, buffer, fields, append=True)
        logger.info(f"💾 Финальное сохранение, всего строк: {saved}")

if __name__ == "__main__":
    data = load_data(INPUT_JSON)
    analyze(data, OUTPUT_CSV)
    logger.info("✅ Анализ завершён.")
