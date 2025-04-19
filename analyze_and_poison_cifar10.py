# process_all_attacks_csv.py  (версия «2 CSV»)
import os, csv, json, random, logging, traceback
from collections import OrderedDict, defaultdict
from tqdm import tqdm
from PIL import Image
import numpy as np
from analysis.full_feature_extractor import extract_all_features

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

SUPPORTED_DATASETS  = ["cifar10", "gtsrb"]
TRIGGER_POSITIONS   = ["bottom_right", "top_left", "center"]
TRIGGERS_DIR        = "triggers"
CHUNK_SIZE          = 500                                    # сколько строк буфера держим в памяти
OUT_DIR             = "data/poisoned"                        # куда кладём все csv
os.makedirs(OUT_DIR, exist_ok=True)


# ───────────── helpers ────────────────────────────────────────────────────────
def load_triggers_by_type(t_type):
    return [os.path.join(TRIGGERS_DIR, f) for f in sorted(os.listdir(TRIGGERS_DIR))
            if f.lower().startswith(f"{t_type}_trigger") and f.lower().endswith((".jpg", ".png"))]

def apply_trigger(image, trig_path, pos="bottom_right", alpha=0.2):
    img = np.asarray(image).copy()
    h, w  = img.shape[:2]
    trig  = Image.open(trig_path).convert("RGB")
    trig  = trig.resize((int(w * .2), int(h * .2)))          # 20 % по ширине/высоте
    tw,th = trig.size

    if   pos == "bottom_right": x,y = w-tw, h-th
    elif pos == "top_left":     x,y = 0,0
    elif pos == "center":       x,y = (w-tw)//2, (h-th)//2
    else:                       x,y = w-tw, h-th

    t_np = np.asarray(trig)
    if "blended" in trig_path.lower():
        img[y:y+th, x:x+tw] = ((1-alpha)*img[y:y+th, x:x+tw] + alpha*t_np).astype(np.uint8)
    else:
        img[y:y+th, x:x+tw] = t_np
    return Image.fromarray(img), (x, y)

def parse_image_info(path):
    """'…/6_3308.jpg' -> ('6', 3308)"""
    name  = os.path.splitext(os.path.basename(path))[0]
    parts = name.split("_")
    try:                       return parts[0], int(parts[-1])
    except Exception:          return "unknown", -1

def collect_clean_images(root):
    res=[]
    for cls in os.listdir(root):
        d=os.path.join(root,cls)
        if not os.path.isdir(d):        continue
        for f in os.listdir(d):
            if f.lower().endswith(".jpg"):
                res.append((cls,f,os.path.join(d,f)))
    return res

def csv_header(path):
    """Вернёт (header, done_paths) если CSV существует"""
    if not os.path.exists(path):
        return None,set()
    done=set()
    with open(path,newline='',encoding='utf-8') as f:
        r=csv.DictReader(f)
        for row in r:
            done.add(row["path"])
        return r.fieldnames,done

def summarise_anoms(anoms:dict):
    """{'fft_peak':[...]}  →  {'fft_peak_cnt':3,'fft_peak_max':123.4,…}"""
    out=OrderedDict()
    for a_type,lst in anoms.items():
        if not lst:       continue
        scores=[a["score"] for a in lst if isinstance(a,dict) and "score" in a]
        out[f"{a_type}_cnt"]=len(scores)
        out[f"{a_type}_max"]=max(scores) if scores else 0
    return out

def flatten_anoms(base:dict, anoms:dict):
    """Разворачиваем все срабатывания в отдельные строки"""
    rows=[]
    for a_type,lst in anoms.items():
        for rec in lst:
            row = {**base,
                   "anomaly_type":a_type,
                   "x":rec.get("x",-1),
                   "y":rec.get("y",-1),
                   "score":rec.get("score",0)}
            for mk,mv in rec.get("meta",{}).items():
                row[f"meta_{mk}"]=mv
            rows.append(row)
    return rows


# ───────────── основной цикл по датасету ──────────────────────────────────────
def process_ds(ds, clean_root, poisoned_root,
               csv_summary_out, csv_anoms_out,
               N="all", use_gpu=True):

    os.makedirs(poisoned_root, exist_ok=True)
    imgs = collect_clean_images(clean_root)
    if str(N).lower()!="all":
        imgs = imgs[:min(int(N),len(imgs))]
    logger.info(f"🔍 {ds}: будет обработано {len(imgs)} изображений")

    # Загружаем существующие CSV, если продолжение
    sum_header, done_paths  = csv_header(csv_summary_out)
    ano_header, _           = csv_header(csv_anoms_out)

    buf_sum, buf_ano = [], []

    # локальный flush
    def flush():
        def _write(path, header, buffer):
            if not buffer: return header
            mode='a' if os.path.exists(path) else 'w'
            with open(path,mode, newline='', encoding='utf-8') as f:
                w=csv.DictWriter(f, fieldnames=header or buffer[0].keys())
                if mode=='w': w.writeheader()
                w.writerows(buffer)
            buffer.clear()
            return w.fieldnames

        nonlocal sum_header, ano_header
        sum_header = _write(csv_summary_out, sum_header, buf_sum)
        ano_header = _write(csv_anoms_out,  ano_header, buf_ano)

    # ─── основной проход ────────────────────────────────────
    for cls,fname,path in tqdm(imgs,desc=f"[{ds}]"):
        if path in done_paths:       continue
        try:
            img = Image.open(path).convert("RGB")
            class_id,idx = parse_image_info(path)

            # clean
            m_clean, a_clean = extract_all_features(img, use_gpu=use_gpu)
            base_clean = {
                "dataset":ds,"class":class_id,"index":idx,"path":path,
                "type":"clean","attack_type":"","trigger_position":"",
                "trigger_x":-1,"trigger_y":-1,
            }

            buf_sum.append({**base_clean, **m_clean, **summarise_anoms(a_clean)})
            buf_ano.extend(flatten_anoms(base_clean, a_clean))

            # poisoned
            for attack in ["badnet","trojan","blended"]:
                for trig_path,pos in zip(load_triggers_by_type(attack)[:3], TRIGGER_POSITIONS):
                    try:
                        p_img,(tx,ty)=apply_trigger(img,trig_path,pos)
                        out_dir=os.path.join(poisoned_root,cls)
                        os.makedirs(out_dir,exist_ok=True)
                        out_name=fname.replace(".jpg",f"_poisoned_{attack}_{pos}.jpg")
                        out_path=os.path.join(out_dir,out_name)
                        p_img.save(out_path)

                        m_p,a_p = extract_all_features(p_img,use_gpu=use_gpu)
                        base_p = {
                            "dataset":ds,"class":class_id,"index":idx,"path":out_path,
                            "type":"poisoned","attack_type":attack,
                            "trigger_position":pos,"trigger_x":tx,"trigger_y":ty,
                        }
                        buf_sum.append({**base_p, **m_p, **summarise_anoms(a_p)})
                        buf_ano.extend(flatten_anoms(base_p, a_p))

                    except Exception:
                        logger.exception(f"[!] триггер {attack}-{pos} ({path})")

        except Exception:
            logger.exception(f"[!] ошибка при {path}")

        if len(buf_sum)>=CHUNK_SIZE or len(buf_ano)>=CHUNK_SIZE:
            flush(); logger.info("💾 промежуточный flush")

    flush()
    logger.info(f"✅ summary → {csv_summary_out}")
    logger.info(f"✅ anomalies → {csv_anoms_out}")


# ───────────── запускаем для всех поддерживаемых датасетов ────────────────────
if __name__=="__main__":
    for ds in SUPPORTED_DATASETS:
        process_ds(
            ds,
            clean_root    =f"data/clean/{ds}",
            poisoned_root =f"data/poisoned/{ds}",
            csv_summary_out =f"{OUT_DIR}/features_{ds}_summary.csv",
            csv_anoms_out   =f"{OUT_DIR}/features_{ds}_anomalies.csv",
            N="all",
            use_gpu=True
        )
