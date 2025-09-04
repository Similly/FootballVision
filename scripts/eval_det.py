#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class-agnostic evaluation of YOLO-normalized detections vs GT (pixel space).
Unterstützt sowohl flache als auch SNMOT-Ordnerstruktur.

Beispiele
---------
# 1) Deine Struktur:
# GT:   data/tracking/challenge/SNMOT-xxx/labels/<frame>.txt
# Pred: eval_chal/outputDet/SNMOT-xxx/labels/<frame>.txt
python eval_det_agnostic.py \
  --gt_dir data/tracking/challenge \
  --pred_dir eval_chal/outputDet \
  --width 1920 --height 1080

# 2) Mit Bildgrößen pro Sequenz (liest <images_root>/<seq>/(img1|images)/*.jpg)
python eval_det_agnostic.py \
  --gt_dir data/tracking/challenge \
  --pred_dir eval_chal/outputDet \
  --images_root data/tracking/challenge
"""
import os, glob, argparse
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import numpy as np

# progress bar (tqdm optional)
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):  # fallback ohne Progressbar
        return x

try:
    import cv2
except ImportError:
    cv2 = None

def _pred_label_path(pred_root, seq_name, stem):
    # probiere mehrere Varianten
    candidates = [
        os.path.join(pred_root, seq_name, 'labels',   stem + '.txt'),              # SNMOT-021
        os.path.join(pred_root, seq_name.replace('SNMOT-', ''), 'labels', stem + '.txt'),  # 021
        os.path.join(pred_root, seq_name.replace('-', '_'), 'labels', stem + '.txt'),      # SNMOT_021
        os.path.join(pred_root, seq_name, 'labels5', stem + '.txt'),               # labels5 fallback
        os.path.join(pred_root, seq_name.replace('SNMOT-', ''), 'labels5', stem + '.txt'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]  # fallback (nicht existent -> zählt als "kein Pred")

# ---------- IO & Geometrie ----------

def parse_yolo_line(line: str):
    """Erlaubt: 'cls xc yc w h' ODER 'cls conf xc yc w h' (alle normiert)."""
    p = line.strip().split()
    if len(p) == 5:
        c, xc, yc, w, h = p
        return int(c), 1.0, float(xc), float(yc), float(w), float(h)
    if len(p) == 6:
        c, conf, xc, yc, w, h = p
        return int(c), float(conf), float(xc), float(yc), float(w), float(h)
    raise ValueError(f"Unsupported line with {len(p)} fields: {line}")

def load_labels_txt(path: str):
    """Liest eine YOLO-Labeldatei. Rückgabe ggf. []."""
    if not os.path.exists(path): return []
    out = []
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(parse_yolo_line(ln))
    return out

def yolo_norm_to_xywh_pixels(xc, yc, w, h, W, H):
    """Normierte Center-Form -> Pixel (xywh, Top-Left)."""
    pw = max(0.0, w * W)
    ph = max(0.0, h * H)
    px = max(0.0, xc * W - 0.5 * pw)
    py = max(0.0, yc * H - 0.5 * ph)
    px = min(px, max(0.0, W - 1.0))
    py = min(py, max(0.0, H - 1.0))
    return px, py, pw, ph

def xywh_to_xyxy(b):
    x, y, w, h = b
    return np.array([x, y, x+w, y+h], dtype=np.float32)

def iou_xyxy(a, b):
    """IoU für [x1,y1,x2,y2]."""
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, xB-xA), max(0.0, yB-yA)
    inter = iw * ih
    if inter <= 0: return 0.0
    areaA = max(0.0, (a[2]-a[0])) * max(0.0, (a[3]-a[1]))
    areaB = max(0.0, (b[2]-b[0])) * max(0.0, (b[3]-b[1]))
    return float(inter / (areaA + areaB - inter + 1e-9))

def find_image_size_for_frame(stem: str, images_dir: str) -> Optional[Tuple[int,int]]:
    """Liest (W,H) aus Bilddatei, wenn vorhanden."""
    if cv2 is None: return None
    for ext in (".jpg",".png",".jpeg",".JPG",".PNG",".JPEG"):
        p = os.path.join(images_dir, stem+ext)
        if os.path.exists(p):
            img = cv2.imread(p)
            if img is not None:
                h, w = img.shape[:2]
                return w, h
    return None

def find_seq_images_dir(images_root: Optional[str], seq_name: str) -> Optional[str]:
    """Sucht <images_root>/<seq>/(img1|images)."""
    if not images_root: return None
    cands = [
        os.path.join(images_root, seq_name, "img1"),
        os.path.join(images_root, seq_name, "images"),
    ]
    for p in cands:
        if os.path.isdir(p):
            return p
    return None

# ---------- Metriken ----------

def voc_ap(rec, prec):
    """AP über Präzisionshülle (kontinuierlich)."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size-1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx+1]-mrec[idx]) * mpre[idx+1]))

def evaluate_agnostic_at_iou(gt_by_key, pred_by_key, iou_thr: float):
    """
    gt_by_key: dict[key] -> list[xyxy]
    pred_by_key: dict[key] -> list[(xyxy, score)]
    key = '<seq>/<frame_stem>' sorgt für Sequenz-Trennung.
    """
    all_preds = []
    for k, items in pred_by_key.items():
        for (box, score) in items:
            all_preds.append((k, box, float(score)))
    all_preds.sort(key=lambda x: -x[2])
    npos = sum(len(v) for v in gt_by_key.values())

    if npos == 0:
        tp = np.zeros(len(all_preds), dtype=np.float32)
        fp = np.ones(len(all_preds), dtype=np.float32) if len(all_preds) else np.array([0.0])
        tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
        rec = tp_c / 1
        prec = tp_c / np.maximum(1, tp_c + fp_c)
        return 0.0, prec, rec, 0, int(fp_c[-1] if len(fp_c) else 0), 0

    gt_matched = {k: np.zeros(len(gt_by_key.get(k, [])), dtype=bool) for k in gt_by_key.keys()}
    tp = np.zeros(len(all_preds), dtype=np.float32)
    fp = np.zeros(len(all_preds), dtype=np.float32)

    for i, (k, pbox, _) in enumerate(all_preds):
        gts = gt_by_key.get(k, [])
        if not gts:
            fp[i] = 1.0
            continue
        ious = np.array([iou_xyxy(pbox, g) for g in gts], dtype=np.float32)
        j = int(np.argmax(ious)) if len(ious) else -1
        best = ious[j] if j >= 0 else 0.0
        if best >= iou_thr and not gt_matched[k][j]:
            tp[i] = 1.0
            gt_matched[k][j] = True
        else:
            fp[i] = 1.0

    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    rec = tp_c / max(1, npos)
    prec = tp_c / np.maximum(1, tp_c + fp_c)
    ap = voc_ap(rec, prec)
    return ap, prec, rec, int(tp_c[-1] if len(tp_c) else 0), int(fp_c[-1] if len(fp_c) else 0), npos

# ---------- Loader für flach/verschachtelt ----------

def collect_pairs(gt_dir: str, pred_dir: str) -> List[Tuple[str, str, str, str]]:
    """
    Liefert Liste von (seq_name, frame_stem, gt_path, pred_path).
    Unterstützt:
      - flach: gt_dir/*.txt  & pred_dir/*.txt
      - SNMOT: gt_dir/SNMOT-*/(labels|labels5)/*.txt
               pred_dir/SNMOT-*/(labels|labels5)/*.txt
    """
    pairs: List[Tuple[str, str, str, str]] = []

    # Fall A: flach
    flat = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))
    if flat:
        for gt_path in flat:
            stem = os.path.splitext(os.path.basename(gt_path))[0]
            seq = ""  # keine Sequenz
            pred_path = os.path.join(pred_dir, stem + ".txt")
            pairs.append((seq, stem, gt_path, pred_path))
        return pairs

    # Fall B: SNMOT-Struktur
    seq_dirs = sorted([p for p in glob.glob(os.path.join(gt_dir, "SNMOT-*")) if os.path.isdir(p)])
    if not seq_dirs:
        raise FileNotFoundError(f"Keine GT .txt gefunden in '{gt_dir}'. "
                                f"Erwartet entweder *.txt direkt oder SNMOT-*/labels/*.txt")

    for seq_path in seq_dirs:
        seq_name = os.path.basename(seq_path)
        gt_label_dir = None
        for lbl in ("labels", "labels5"):
            cand = os.path.join(seq_path, lbl)
            if os.path.isdir(cand):
                gt_label_dir = cand
                break
        if gt_label_dir is None:
            continue

        gt_txts = sorted(glob.glob(os.path.join(gt_label_dir, "*.txt")))
        if not gt_txts:
            continue

        for gt_path in gt_txts:
            stem = os.path.splitext(os.path.basename(gt_path))[0]
            pred_path = _pred_label_path(pred_dir, seq_name, stem)
            pairs.append((seq_name, stem, gt_path, pred_path))

    return pairs

# ---------- Pipeline ----------

def evaluate_detection_agnostic(gt_dir: str,
                                pred_dir: str,
                                width: Optional[int],
                                height: Optional[int],
                                images_root: Optional[str],
                                iou_thresholds: List[float],
                                coco_range: Tuple[float,float,float]):

    gt_by_key = defaultdict(list)   # key = '<seq>/<stem>' (seq leer für flachen Modus)
    pred_by_key = defaultdict(list)

    # pro-Sequenz Bildordner merken, falls images_root angegeben
    seq2imgdir: Dict[str, Optional[str]] = {}

    pairs = collect_pairs(gt_dir, pred_dir)
    if not pairs:
        raise FileNotFoundError(f"Keine passenden GT-Dateien gefunden unter {gt_dir}")

    missing_pred = 0

    for seq, stem, gt_path, pred_path in tqdm(pairs, desc="Lade & matche Frames", unit="frame"):
        # Bildgröße bestimmen
        if images_root:
            if seq not in seq2imgdir:
                seq2imgdir[seq] = find_seq_images_dir(images_root, seq) if seq else images_root
            img_dir = seq2imgdir[seq]
            if img_dir:
                wh = find_image_size_for_frame(stem, img_dir)
            else:
                wh = None
        else:
            wh = None

        if wh is not None:
            W, H = wh
        else:
            if width is None or height is None:
                raise ValueError("Entweder --width/--height angeben ODER --images_root (mit img1/images je Sequenz).")
            W, H = width, height

        # GT laden (klassenagnostisch)
        gt_items = load_labels_txt(gt_path)
        key = f"{seq}/{stem}" if seq else stem
        for _c, _conf, xc, yc, w, h in gt_items:
            x, y, ww, hh = yolo_norm_to_xywh_pixels(xc, yc, w, h, W, H)
            gt_by_key[key].append(xywh_to_xyxy([x, y, ww, hh]))

        # Pred laden (kann fehlen)
        pred_items = load_labels_txt(pred_path)
        if not pred_items:
            missing_pred += 1
        for _c, conf, xc, yc, w, h in pred_items:
            x, y, ww, hh = yolo_norm_to_xywh_pixels(xc, yc, w, h, W, H)
            pred_by_key[key].append((xywh_to_xyxy([x, y, ww, hh]), float(conf)))

    # COCO IoU-Range
    lo, hi, step = coco_range
    coco_thrs = np.arange(lo, hi + 1e-9, step)

    # Einzel-Thresholds
    res = {}
    for thr in iou_thresholds:
        ap, prec, rec, tp, fp, npos = evaluate_agnostic_at_iou(gt_by_key, pred_by_key, float(thr))
        res[f"AP@{thr:.2f}"] = ap
        res[f"TP@{thr:.2f}"] = tp
        res[f"FP@{thr:.2f}"] = fp
        res[f"GT@{thr:.2f}"] = npos

    # COCO Mittel
    ap_list = []
    for thr in coco_thrs:
        ap, *_ = evaluate_agnostic_at_iou(gt_by_key, pred_by_key, float(thr))
        ap_list.append(ap)
    res["AP@[0.50:0.95]"] = float(np.mean(ap_list)) if ap_list else 0.0

    res["_pairs"] = len(pairs)
    res["_frames_without_pred"] = missing_pred
    return res

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Class-agnostic eval (YOLO normalized -> pixel).")
    ap.add_argument("--gt_dir", required=True, help="GT-Root (flach: *.txt; SNMOT: SNMOT-*/labels/*.txt)")
    ap.add_argument("--pred_dir", required=True, help="Pred-Root (flach: *.txt; SNMOT: <seq>/labels/*.txt)")
    ap.add_argument("--width", type=int, default=None, help="Global image width (falls kein --images_root)")
    ap.add_argument("--height", type=int, default=None, help="Global image height (falls kein --images_root)")
    ap.add_argument("--images_root", type=str, default=None, help="Root, das pro Sequenz (img1|images) enthält")
    ap.add_argument("--iou_thresholds", type=float, nargs="+", default=[0.50, 0.75])
    ap.add_argument("--coco_iou_range", type=float, nargs=3, default=[0.50, 0.95, 0.05])
    args = ap.parse_args()

    res = evaluate_detection_agnostic(
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        width=args.width,
        height=args.height,
        images_root=args.images_root,
        iou_thresholds=args.iou_thresholds,
        coco_range=tuple(args.coco_iou_range)
    )

    print("\nClass-agnostic results:")
    for k in sorted(res.keys()):
        if k.startswith(("AP@", "AP@[")):
            print(f"{k}: {res[k]:.4f}")
        elif not k.startswith("_"):
            print(f"{k}: {res[k]}")
    # kleine Diagnose
    print(f"\n[diag] matched frames: {res.get('_pairs', 0)} | frames without pred: {res.get('_frames_without_pred', 0)}")

if __name__ == "__main__":
    main()
