# fast_analyser.py

import os, json, random, logging, traceback
from tqdm import tqdm
from PIL import Image
import numpy as np
from analysis.full_feature_extractor import extract_all_features

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# ─── ПАРАМЕТРЫ ────────────────────────────────────────────────────────────────
DATASET          = "cifar10"          # "cifar10" или "gtsrb"
NUM_SAMPLES      = 100                # clean + poisoned
USE_GPU          = True               # CUDA?
TRIGGER_PATH     = "triggers/badnet_trigger_01.png"
TRIGGER_POSITION = "bottom_right"     # фиксируем позицию
ROOT_CLEAN       = f"data/clean/{DATASET}"
ROOT_POISONED    = f"data/poisoned/{DATASET}"
OUT_JSON         = f"data/poisoned/features_{DATASET}_simple.json"
ATTACK_TYPE      = "badnet"           # единственная атака
# ------------------------------------------------------------------------------

# ─── вспомогательные функции ──────────────────────────────────────────────────
def apply_badnet_trigger(img: Image.Image, pos=TRIGGER_POSITION):
    arr      = np.array(img).copy()
    h, w     = arr.shape[:2]

    trig_img = Image.open(TRIGGER_PATH).convert("RGB")
    trig_img = trig_img.resize((int(w * 0.2), int(h * 0.2)))   # 20 % стороны
    t_w, t_h = trig_img.size
    trig_np  = np.array(trig_img)

    if   pos == "bottom_right":  x, y = w - t_w, h - t_h
    elif pos == "top_left":      x, y = 0, 0
    elif pos == "center":        x, y = (w - t_w)//2, (h - t_h)//2
    else:                        x, y = w - t_w, h - t_h

    arr[y:y+t_h, x:x+t_w] = trig_np
    return Image.fromarray(arr), (x, y)

def parse_image_info(p):
    stem = os.path.splitext(os.path.basename(p))[0].split("_")
    try:  return stem[0], int(stem[-1])
    except: return "unknown", -1

def collect_first_n(root, n):
    pool=[]
    for cls in sorted(os.listdir(root)):
        d = os.path.join(root, cls)
        if not os.path.isdir(d): continue
        pool += [(cls,f,os.path.join(d,f))
                 for f in os.listdir(d) if f.lower().endswith(".jpg")]
    random.shuffle(pool)
    return pool[:n]

# ─── основной процесс ─────────────────────────────────────────────────────────
def run():
    os.makedirs(ROOT_POISONED, exist_ok=True)
    paths = collect_first_n(ROOT_CLEAN, NUM_SAMPLES)
    log.info(f"Старт: {DATASET}, изображений={len(paths)}, GPU={USE_GPU}")

    rows=[]
    for cls, fname, full in tqdm(paths, desc=f"[{DATASET}]"):
        try:
            img_clean = Image.open(full).convert("RGB")
            cid, idx  = parse_image_info(full)

            # ---- clean ----
            met_c, anom_c = extract_all_features(img_clean, use_gpu=USE_GPU)
            rows.append({
                "dataset": DATASET, "class": cid, "index": idx, "path": full,
                "type": "clean", "attack_type": "",
                "trigger_position": "", "trigger_x": -1, "trigger_y": -1,
                "metrics": met_c, "anomalies": anom_c
            })

            # ---- poisoned ----
            img_p,(tx,ty) = apply_badnet_trigger(img_clean)
            out_dir = os.path.join(ROOT_POISONED, cls); os.makedirs(out_dir, exist_ok=True)
            out_fn  = fname.replace(".jpg", f"_poisoned_{ATTACK_TYPE}_{TRIGGER_POSITION}.jpg")
            out_fp  = os.path.join(out_dir, out_fn)
            img_p.save(out_fp)

            met_p, anom_p = extract_all_features(img_p, use_gpu=USE_GPU)
            rows.append({
                "dataset": DATASET, "class": cid, "index": idx, "path": out_fp,
                "type": "poisoned", "attack_type": ATTACK_TYPE,
                "trigger_position": TRIGGER_POSITION, "trigger_x": tx, "trigger_y": ty,
                "metrics": met_p, "anomalies": anom_p
            })

        except Exception as e:
            log.error(f"[ERR] {full}: {e}")
            traceback.print_exc()

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    log.info(f"✅ JSON сохранён: {OUT_JSON}  (строк: {len(rows)})")

# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()
