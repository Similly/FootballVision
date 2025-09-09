import os
import re
import sys
import cv2
import time as ti
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import models
from torchvision import transforms as T
from torchvision.models import ResNet34_Weights
from pathlib import Path
from collections import Counter
from ultralytics import YOLO
from sklearn.cluster import KMeans
from boxmot import BotSort, BoostTrack
from collections import defaultdict, deque

# === OCR-Parameter ===
OCR_FRAME_GAP = 1                    # Keyframe-Sampling: alle N Frames
MIN_PLAYER_H = 50                    # Mindesthöhe Spielerbox für OCR
AREA_GROWTH_TRIG = 1.4               # OCR triggern, wenn Boxfläche stark wächst
CONF_MIN = 0.60                      # Mindestkonfidenz (nach Postprocess)
LEAKY_DECAY = 0.001                  # Leaky pro Frame (kleiner Wert)

DRAW_ROI = True
ROI_COLORS = {"torso": (0, 255, 255)}  # BGR
ROI_THICK = 2

LEGIBILITY_ENABLED = True
LEG_THR = 0.75
LEG_MIN_H, LEG_MIN_W = 40, 40

USE_PARSEQ = True
PARSEQ_CKPT = "ocrModels/epoch=4-step=665-val_accuracy=96.5800-val_NED=97.8600.ckpt"

def make_parseq_transform(img_size, rotation=0, augment=False):
    trans = []
    if rotation:
        trans.append(lambda img: img.rotate(rotation, expand=True))
    trans.extend([
        T.Resize(img_size, T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),  # <-- fix
    ])
    return T.Compose(trans)

# ImageNet-Normalisierung für ResNet
_leg_tf = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def _to_batch_rgb_tensor(bgr_np, device):
    rgb = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB)
    ten = _leg_tf(rgb).unsqueeze(0).to(device)  # [1,3,224,224]
    return ten

@torch.inference_mode()
def legibility_score(roi_bgr, leg_model, device):
    if roi_bgr is None or roi_bgr.size == 0:
        return 0.0
    h, w = roi_bgr.shape[:2]
    if h < LEG_MIN_H or w < LEG_MIN_W:
        return 0.0
    x = _to_batch_rgb_tensor(roi_bgr, device)
    prob = float(leg_model(x).sigmoid().item() if leg_model.training else leg_model(x).item())
    # (Falls dein forward schon sigmoid macht, ist .sigmoid() überflüssig. Oben abgefangen.)
    return prob

def is_legible(roi_bgr, leg_model, device, thr=LEG_THR):
    p = legibility_score(roi_bgr, leg_model, device)
    return (p >= thr), p

# === ResNet34 für Legibility-Klassifikation ===
class LegibilityClassifier34(nn.Module):
    def __init__(self, ckpt_path=None, finetune=False, init_from_imagenet=False, device="cpu"):
        super().__init__()
        base_weights = ResNet34_Weights.IMAGENET1K_V1 if init_from_imagenet else None
        self.model_ft = models.resnet34(weights=base_weights) 

        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, 1)

        if finetune:
            for p in self.model_ft.parameters(): 
                p.requires_grad = False
            for p in self.model_ft.layer4.parameters():
                p.requires_grad = True
            for p in self.model_ft.fc.parameters():
                p.requires_grad = True

        if ckpt_path:
            state = torch.load(ckpt_path, map_location=device)
            if isinstance(state, dict) and "state_dict" in state: 
                state = state["state_dict"]
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            state = {k.replace("module.", ""): v for k, v in state.items()}
            try:
                self.load_state_dict(state, strict=True)
            except RuntimeError:
                self.load_state_dict(state, strict=False)

    def forward(self, x):
        # gibt bereits Sigmoid-Probability zurück (0..1)
        x = self.model_ft(x)
        return torch.sigmoid(x)

# === Team-Cluster-Funktionen ===
def get_grass_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    return hsv, mask


def extract_kit_color(player_img, grass_hsv_val):
    if player_img is None or player_img.size == 0:
        return np.array([0,0,0])

    hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
    # Gras-Maske basierend auf Hintergrund-Farbton
    lower = np.array([grass_hsv_val - 10, 40, 40])
    upper = np.array([grass_hsv_val + 10, 255, 255])
    grass_mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.bitwise_not(grass_mask)
    # nur obere Hälfte (Trikot) nutzen
    h, w = mask.shape
    mask[int(0.5*h):, :] = 0
    if np.count_nonzero(mask) == 0:
        return np.array([0,0,0])
    mean_bgr = cv2.mean(player_img, mask=mask)[:3]
    return np.array(mean_bgr)

track_history = defaultdict(lambda: deque(maxlen=50))
def get_dominant_team(history_deque):
    if not history_deque:
        return None
    # Most common label
    return max(set(history_deque), key=history_deque.count)

# === OCR Helfer ===
def variance_of_laplacian(img_gray):
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()

def preprocess_roi(roi_bgr):
    # Upscale leicht, Graustufen + Histogramm-Equalize, adaptive Threshold
    H, W = roi_bgr.shape[:2]
    scale = 2 if max(H, W) < 200 else 1
    if scale != 1:
        roi_bgr = cv2.resize(roi_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    # leichte adaptive Binarisierung (nur als Kontrast-Booster; PaddleOCR kann auch Graustufen)
    bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 21, 5)
    # Zurück in 3-Kanal (PaddleOCR akzeptiert np.ndarray; 3-Kanal ist robust)
    bin_bgr = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
    return bin_bgr, gray

def preprocess_roi_parseq(roi_bgr):
    H, W = roi_bgr.shape[:2]
    scale = 2 if max(H, W) < 200 else 1
    if scale != 1:
        roi_bgr = cv2.resize(roi_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # Kontrast sanft anheben, keine binäre Schwelle:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    # Histogramm-Stretch
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    # zurück nach RGB (3-Kanal, identische Kanäle)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return rgb


def jersey_rois_with_abs(frame, x1, y1, x2, y2):
    H = max(0, y2 - y1)
    W = max(0, x2 - x1)
    if H < MIN_PLAYER_H or W < 40:
        return []

    # relative Koordinaten in der Spielerbox
    # Rücken: mittleres bis unteres Drittel der oberen Körperhälfte
    ry1, ry2 = int(y1 + 0.15 * H), int(y1 + 0.55 * H)
    rx1, rx2 = int(x1 + 0.15 * W), int(x1 + 0.85 * W)
    back_box = (rx1, ry1, rx2, ry2)

    # Clampen an Frame-Grenzen
    hF, wF = frame.shape[:2]
    def clamp_box(b):
        x1b, y1b, x2b, y2b = b
        x1b = max(0, min(wF-1, x1b)); x2b = max(0, min(wF-1, x2b))
        y1b = max(0, min(hF-1, y1b)); y2b = max(0, min(hF-1, y2b))
        if x2b <= x1b or y2b <= y1b:
            return None
        return (x1b, y1b, x2b, y2b)

    out = []
    for box, tag in [(back_box, "torso")]:
        c = clamp_box(box)
        if c is None:
            continue
        cx1, cy1, cx2, cy2 = c
        roi = frame[cy1:cy2, cx1:cx2]
        out.append((roi, (cx1, cy1, cx2, cy2), tag))
    return out

def read_number_from_roi(roi_bgr, ocr, conf_min=0.50):  # conf_min leicht gelockert
    H, W = roi_bgr.shape[:2]
    ar = W / max(1, H)
    if not (0.2 <= ar <= 5.0):  # ROI_AR_MAX auf 5.0 erweitert
        return None

    if USE_PARSEQ:
        pre_img = preprocess_roi_parseq(roi_bgr)   # statt preprocess_roi(...)
    else:
        pre_img, gray = preprocess_roi(roi_bgr)
        if variance_of_laplacian(gray) < 50.0:  # LAP_VAR_TH von 80 -> 50
            return None

    texts, scores, boxes, polys = [], [], None, None

    text, conf = parseq_infer_text(pre_img, ocr, device)  # device = global
    texts  = [text]
    scores = [conf]
    boxes  = None
    polys  = None
    #print(f"[F{frame_idx} T{track_id} ROI=({rx1},{ry1},{rx2},{ry2})] PARSeq: {text} (conf={conf:.2f})")

    if len(texts) == 0:
        return None

    roi_cx, roi_cy = W * 0.5, H * 0.5

    def center_from_box_or_poly(i):
        if boxes is not None and i < len(boxes):
            x1, y1, x2, y2 = boxes[i]
            return (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if polys is not None and i < len(polys):
            ps = polys[i]
            xs = [p[0] for p in ps]; ys = [p[1] for p in ps]
            return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
        return roi_cx, roi_cy

    best = None
    for i, (txt, rec_conf) in enumerate(zip(texts, scores)):
        digits = re.sub(r"[^0-9]", "", str(txt))
        if len(digits) == 0 or len(digits) > 2:
            continue
        val = int(digits)
        if not (1 <= val <= 99):
            continue

        cx, cy = center_from_box_or_poly(i)
        dist = np.hypot(cx - roi_cx, cy - roi_cy) / (0.5 * np.hypot(W, H) + 1e-6)
        geom = max(0.7, 1.0 - 0.5 * dist)  # 0.7..1.0
        conf = float(rec_conf) * float(geom)

        if best is None or conf > best[1]:
            best = (val, conf)

    if best is None or best[1] < conf_min:
        return None
    return best  # (nummer, konfidenz)

def load_parseq_from_ckpt(ckpt_path, device):
    import string as _s
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
    # Keys vereinheitlichen
    fixed = {}
    for k, v in sd.items():
        k = k.replace("module.", "").replace("_orig_mod.", "")
        if not k.startswith("model."):
            k = "model." + k
        fixed[k] = v

    # Architektur & Längen aus dem CKPT ableiten
    pq = fixed["model.pos_queries"]           # [1, n_queries, d_model]
    n_queries = pq.shape[1]
    d_model   = pq.shape[2]
    arch = "parseq_tiny" if d_model <= 192 else "parseq"
    max_len = int(n_queries - 1)

    head_out  = fixed["model.head.weight"].shape[0]
    embed_num = fixed["model.text_embed.embedding.weight"].shape[0]

    # Charset bestimmen
    hp = ckpt.get("hyper_parameters", {})
    if head_out == 11:
        charset_str = _s.digits  # "0123456789"
    else:
        charset_str = hp.get("charset") or "".join([c for c in _s.printable if c not in "\t\n\r\x0b\x0c"])

    # Modell instanziieren – mehrere API-Varianten probieren
    def _build(**kw):
        return torch.hub.load('baudm/parseq', arch, pretrained=False,
                              trust_repo=True, max_label_length=max_len, **kw)

    tried = []
    for kw in ({"charset": charset_str},
               {"charset_train": charset_str, "charset_test": charset_str},
               {"charset": charset_str, "charset_train": charset_str, "charset_test": charset_str}):
        try:
            m = _build(**kw)
            if (m.model.head.weight.shape[0] == head_out and
                m.model.text_embed.embedding.weight.shape[0] == embed_num):
                model = m
                break
        except Exception as e:
            tried.append((kw, str(e)))
    else:
        # Fallback: danach Head/Embedding passend umbauen
        model = _build()
        model.model.head = torch.nn.Linear(d_model, head_out, bias=True)
        model.model.text_embed.embedding = torch.nn.Embedding(embed_num, d_model)
        try:
            # Tokenizer auf Ziffern setzen, wenn digits-only
            if head_out == 11:
                model.tokenizer.set_charset(charset_str)
        except Exception:
            try:
                model.tokenizer.charset = charset_str
            except Exception:
                pass

    # Gewichte laden (nicht strikt!)
    missing, unexpected = model.load_state_dict(fixed, strict=False)
    if missing or unexpected:
        print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")

    model = model.to(device).eval()
    tfm = make_parseq_transform(tuple(hp.get("img_size", (32,128))))
    print(f"[PARSeq] arch={arch} d_model={d_model} max_len={max_len} head_out={head_out} embed_num={embed_num}")
    return model, tfm

def parseq_infer_text(bgr_roi, model, device):
        """Spiegelt die Inferenz aus dem Jersey-Repo:
           - RGB + offizieller Transform
           - logits[:, :3, :11] (3 Ziffern; Klassen: 0..9 + <eos>)
           - tokenizer.decode auf den gesliceten Probs
           -> Rückgabe (text, conf)  (conf = Produkt der Token-Prob.)
        """
        if bgr_roi is None or bgr_roi.size == 0:
            return "", 0.0
        rgb = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2RGB)
        x = parseq_tf(Image.fromarray(rgb)).unsqueeze(0).to(device)  # [1,3,H,W]

        logits = model(x)                    # [B, T, C]  (voller Zeichensatz)
        logits = logits[:, :3, :11]          # *** wie im Repo ***
        probs  = F.softmax(logits, dim=-1)   # [B, 3, 11]

        preds, probs_list = model.tokenizer.decode(probs)  # nutzt charset_test=digits
        text = preds[0]
        # Konfidenz: Produkt der drei Stellen
        if isinstance(probs_list, (list, tuple)) and len(probs_list) > 0:
            # probs_list[0] ist Tensor/List der Token-Prob. per Stelle
            conf = float(torch.as_tensor(probs_list[0]).prod().item())
        else:
            conf = float(probs_list)
        return text, conf

# === Track-Level Fusing ===
track_assigned_num = {}                         # tid -> int (aktuelle Entscheidung)
last_ocr_frame  = defaultdict(lambda: -10)
last_box_area   = defaultdict(lambda: 0.0)

VOTE_MIN_TOTAL_CONF = 1.5                       # Mindest-Summe Konfidenz im Track, bevor wir assignen
ONE_DIGIT_PENALTY   = 0.8                       # 1-stellige Nummern etwas abwerten vs. 2-stellig
HYSTERESIS_MARGIN   = 1.2                       # neuer Gewinner braucht >=20% mehr Gewicht als #2
DELTA_MIN = 0.30   # Mindestvorsprung im absoluten Vote-Gewicht
USE_LEAKY           = True

# Aggregatoren
vote_sum   = defaultdict(lambda: defaultdict(float))  # tid -> {nummer: gewicht}
vote_sum_offline = defaultdict(lambda: defaultdict(float))
total_conf = defaultdict(float)                       # tid -> Summe conf über alle votes
num_votes  = defaultdict(int)                         # tid -> Anzahl akzeptierter Votes

def add_vote(tid, num, conf):
    # 2-stellig bevorzugen (Paper-Heuristik)
    w = float(conf) * (1.0 if num >= 10 else ONE_DIGIT_PENALTY)
    vote_sum[tid][int(num)] += w
    vote_sum_offline[tid][int(num)] += w
    total_conf[tid] += float(conf)
    num_votes[tid] += 1

def _top2_votes(d):
    if not d: 
        return None, (None, 0.0)
    items = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
    top1 = items[0]
    top2 = items[1] if len(items) > 1 else (None, 0.0)
    return top1, top2

def maybe_assign_number(tid):
    # erst entscheiden, wenn genug Evidenz da ist
    if total_conf[tid] < VOTE_MIN_TOTAL_CONF:
        return
    top1, top2 = _top2_votes(vote_sum[tid])
    if top1 is None:
        return
    top1_num, top1_w = int(top1[0]), float(top1[1])
    prev = track_assigned_num.get(tid, None)
    # erstmalige Zuweisung
    if prev is None:
        track_assigned_num[tid] = int(top1[0])
        return
    # Hysterese gegen Flattern
    if int(prev) != int(top1[0]):
        prev_w = float(vote_sum[tid].get(int(prev), 0.0))
        # Wechsel nur wenn (i) Verhältnis-Hysterese und (ii) absoluter Vorsprung
        if (top1_w >= HYSTERESIS_MARGIN * prev_w) and (top1_w - prev_w >= DELTA_MIN):
            track_assigned_num[tid] = top1_num

def leaky_decay_all():
    if not USE_LEAKY: 
        return
    # sanftes Vergessen wie bisher, aber auf Votes angewendet
    for tid in list(vote_sum.keys()):
        for n in list(vote_sum[tid].keys()):
            vote_sum[tid][n] *= (1.0 - LEAKY_DECAY)
            if vote_sum[tid][n] < 1e-6:
                del vote_sum[tid][n]
        total_conf[tid] *= (1.0 - LEAKY_DECAY)

# === Parameter & Setup ===
# Video input statt image_dir
if len(sys.argv) < 2:
    print("Usage: python script.py <input_video.mp4>")
    sys.exit(1)
video_path = sys.argv[1]

output_dir = "outputVids/track_mot"
os.makedirs(output_dir, exist_ok=True)

# Video-Capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Fehler: Video {video_path} konnte nicht geöffnet werden.")
    sys.exit(1)

# Video-Parameter
fps = cap.get(cv2.CAP_PROP_FPS) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# YOLO & Device
print(f"MPS Available: {torch.backends.mps.is_available()}, Built: {torch.backends.mps.is_built()}")
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("using device:", device)
model = YOLO("detModels/yolov8n+SoccerNet5class_phase2/weights/best.pt")
model.to(device.type)
print(model.names)

# ersten Frame laden für Gras-Farbe
ret, first_frame = cap.read()
if not ret:
    print("Fehler: Konnte ersten Frame nicht lesen.")
    sys.exit(1)
hsv, grass_mask = get_grass_color(first_frame)
grass_hsv_val = cv2.mean(cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV), mask=grass_mask)[0]
# Zurück zum Frame 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Video-Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = os.path.join(output_dir, f"091_output_video_OCR.mp4")
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Data-Writer
mot_file_path = os.path.join(output_dir, "091_mot_results.txt")
output_rows = []

# Farben (Standard für andere Klassen)
colors = {2:(0,0,255),3:(255,255,0)}  # Ball, Ref
target_classes = [0,1,2,3]

# Tracker-Instanzen pro Klasse initialisieren
if os.environ.get('CUDA_VISIBLE_DEVICES', '').strip().lower() == 'cuda':
    os.environ.pop('CUDA_VISIBLE_DEVICES') 

has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
if has_cuda:
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')      # für PyTorch & YOLO
    mot_device = '0'                      # für BoxMOT
else:
    device = torch.device('cpu')
    mot_device = 'cpu'

print("PyTorch sees CUDA:", torch.cuda.is_available(), "num:", torch.cuda.device_count(), "device:", device)

trackers_dict = {}
for class_id in target_classes:
    if class_id == 0: # Player
        if device.type == 'cuda':
            trackers_dict[class_id] = BotSort(
                reid_weights=Path('reid/osnet_x1_0-stripped.pth'),
                device=mot_device,
                half=True,
                track_high_thresh=0.5,
                track_low_thresh=0.1,
                new_track_thresh=0.6,
                track_buffer=50,
                frame_rate=fps,
                proximity_thresh=0.4,
                appearance_thresh=0.15,
            )
        else:
            trackers_dict[class_id] = BotSort(
                with_reid=False,
                reid_weights=None,
                device=mot_device,
                half=False,
                track_high_thresh=0.5,
                track_low_thresh=0.1,
                new_track_thresh=0.6,
                track_buffer=50,
                frame_rate=fps,
                proximity_thresh=0.4,
            )
    elif class_id == 2: # Ball
        trackers_dict[class_id] = BoostTrack(
            with_reid=False,
            reid_weights=None,
            device=mot_device,
            half=False,
            min_hits=0,
            det_thresh=0.05,
            iou_threshold=0.00,
            min_box_area=1,
        )
    elif class_id in [1,3]: # Ref oder Keeper
        trackers_dict[class_id] = BotSort(
            with_reid=False,
            reid_weights=None,
            device=mot_device,
            half=False,
            track_buffer=50,
            frame_rate=fps,
            new_track_thresh=0.6,
        )
    else:
        print(f"Unbekannte Klasse {class_id}, Tracker nicht initialisiert.")
    print(f"Tracker für Klasse {class_id} initialisiert.")

#OCR init
if USE_PARSEQ:
    parseq_sys, parseq_tf = load_parseq_from_ckpt(PARSEQ_CKPT, device)
    ocr = parseq_sys

# === Legibility init (ResNet34) ===
leg_device = device  # gleiches Device wie YOLO bei dir (aktuell CPU)
leg_model = LegibilityClassifier34()

# optional: Gewichte laden
leg_ckpt = "legibility_resnet34_soccer_20240215.pth"
if os.path.exists(leg_ckpt):
    sd = torch.load(leg_ckpt, map_location=leg_device)
    leg_model.load_state_dict(sd, strict=True)
else:
    print("[Legibility] Warnung: kein Checkpoint gefunden – nur ImageNet-Pretrain.")

leg_model.to(leg_device).eval()

# Main Loop
start_time = ti.time()
detection_time = 0
track_time = 0
ocr_time = 0
ocr_calls = 0         # players where do_ocr==True
ocr_roi_calls = 0     # total ROI recognitions (PARSeq/Paddle calls)
frame_idx = 0
kits_clf = None
left_label = None
old_left_center = None
old_right_center = None

while True:
    frame_time = ti.time()
    frame_ocr_time = 0.0
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    print(f"\n{frame_idx}")
    cv2.putText(frame, f"Frame {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    # Leaky Decay global
    leaky_decay_all()

    # YOLO Inferenz
    t0 = ti.time()
    results = model(frame)
    frame_detection_time = ti.time() - t0
    detection_time += frame_detection_time

    # Detections sammeln
    dets = []
    for box in results[0].boxes:
        x1,y1,x2,y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0]); cls = int(box.cls[0])
        dets.append([x1,y1,x2,y2,conf,cls])
    dets = np.array(dets)

    # Tracking pro Klasse
    t1 = ti.time()
    results_by_class = {}
    for class_id in target_classes:
        class_dets = dets[dets[:,5]==class_id] if dets.size else np.empty((0,6))
        online_targets = trackers_dict[class_id].update(class_dets, frame)
        results_by_class[class_id] = online_targets
    frame_track_time = ti.time() - t1
    track_time += frame_track_time

    # Nicht-Player zeichnen (Ball, Ref)
    for cid in [2, 3]:
        for t in results_by_class[cid]:
            x1, y1, x2, y2, _track_id = map(int, t[:5])  # echte ID ignorieren
            w, h = x2 - x1, y2 - y1
            col = colors[cid]
            shown_id = 1  # Ball/Ref immer 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame, f"C{cid} T{shown_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

            output_rows.append((frame_idx, shown_id, float(x1), float(y1),
                                float(w), float(h), -1.0, str(cid), -1))


    # Keeper (GK) Team-Seiten-Logik  => immer ID=1 ausgeben
    for t in results_by_class[1]:
        x1, y1, x2, y2, _track_id = map(int, t[:5])  # echte ID ignorieren
        w, h = x2 - x1, y2 - y1
        side = 'L' if x1 < width / 2 else 'R'
        col = (0, 255, 0) if side == 'L' else (0, 128, 0)
        shown_id = 1  # GK immer 1 (pro Seite unterscheidbar über 'side')

        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv2.putText(frame, f"GK-{side} ID{shown_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        # 'GK-L'/'GK-R'
        extra_str = f"GK-{side}"  

        output_rows.append((frame_idx, shown_id, float(x1), float(y1),
                            float(w), float(h), -1.0, extra_str, -1))


    # Spieler-Team-Cluster (Klasse 0)
    players = results_by_class[0]
    kit_colors = []
    bboxes = []
    for t in players:
        x1, y1, x2, y2, track_id = map(int, t[:5])
        crop = frame[y1:y2, x1:x2]
        color_vec = extract_kit_color(crop, grass_hsv_val)
        kit_colors.append(color_vec)
        bboxes.append((x1,y1,x2,y2,track_id))

    # Periodisches Refit
    if frame_idx == 1 or frame_idx % 125 == 0:
        if len(kit_colors) >= 2:
            new_clf = KMeans(n_clusters=2, random_state=42).fit(kit_colors)
            new_centers = new_clf.cluster_centers_
            if old_right_center is not None:
                dists = [abs(c[0] - old_right_center[0]) for c in new_centers]
                new_right_idx = int(np.argmin(dists))
                new_left_idx = 1 - new_right_idx
                reordered_centers = np.array([new_centers[new_left_idx], new_centers[new_right_idx]])
                class KitsHolder:
                    def __init__(self, centers): self.cluster_centers_ = np.array(centers)
                    def predict(self, X):
                        arr = np.array(X)
                        d = np.linalg.norm(arr[:, None, :] - self.cluster_centers_[None, :, :], axis=2)
                        return np.argmin(d, axis=1)
                kits_clf = KitsHolder(reordered_centers)
                old_left_center, old_right_center = reordered_centers
            else:
                centers_x = [(x1+x2)/2 for (x1,y1,x2,y2,_) in bboxes]
                labels_ref = new_clf.labels_
                avg_x = [np.mean([cx for cx,lbl in zip(centers_x, labels_ref) if lbl==i]) for i in [0,1]]
                right_idx = int(np.argmax(avg_x))
                left_idx = 1 - right_idx
                reordered_centers = np.array([new_clf.cluster_centers_[left_idx], new_clf.cluster_centers_[right_idx]])
                class KitsHolder:
                    def __init__(self, centers): self.cluster_centers_ = np.array(centers)
                    def predict(self, X):
                        arr = np.array(X)
                        d = np.linalg.norm(arr[:, None, :] - self.cluster_centers_[None, :, :], axis=2)
                        return np.argmin(d, axis=1)
                kits_clf = KitsHolder(reordered_centers)
                old_left_center, old_right_center = reordered_centers
                left_label = 0

    # Zeichne Spieler basierend auf aktuellem Klassifizierer
    if kits_clf is not None and kit_colors:
        labels = kits_clf.predict(kit_colors)
        for lbl,(x1,y1,x2,y2,track_id) in zip(labels, bboxes):
            team_label = 'L' if lbl==left_label else 'R'
            track_history[track_id].append(team_label)
            dom_team = get_dominant_team(track_history[track_id])
            col = (255,0,0) if dom_team=='L' else (0,0,255)

            #OCR-Trigger
            box_w, box_h = x2-x1, y2-y1
            area = float(box_w * box_h)
            do_ocr = False

            if frame_idx - last_ocr_frame[track_id] >= OCR_FRAME_GAP:
                do_ocr = True
            if area > last_box_area[track_id] * AREA_GROWTH_TRIG:
                do_ocr = True
            last_box_area[track_id] = max(last_box_area[track_id], area)

            if do_ocr:
                ocr_calls += 1
                t2 = ti.time()
                # 0) Player-Crop + Legibility auf Player (Paper-Logik)
                player_crop = frame[y1:y2, x1:x2]
                leg_ok, leg_p = (True, 1.0)
                if LEGIBILITY_ENABLED:
                    leg_ok, leg_p = is_legible(player_crop, leg_model, leg_device, thr=LEG_THR)

                # Legibility-Score genau EINMAL unter der Player-Box anzeigen
                if DRAW_ROI and LEGIBILITY_ENABLED:
                    leg_col = (0, 255, 0) if leg_ok else (0, 0, 255)
                    leg_y = min(frame.shape[0] - 4, y2 + 16)
                    cv2.putText(frame, f"leg {leg_p:.2f}",
                                (x1, leg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, leg_col, 1)

                got = False
                if leg_ok:
                    # 1) Torso-ROI(s) bestimmen
                    for roi_img, (rx1, ry1, rx2, ry2), tag in jersey_rois_with_abs(frame, x1, y1, x2, y2):
                        # ROI zeichnen, weil wir jetzt OCR versuchen
                        if DRAW_ROI:
                            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), ROI_COLORS.get(tag, (200,200,200)), ROI_THICK)

                        # 2) PARSeq/Paddle auf Torso-ROI
                        ocr_roi_calls += 1
                        out = read_number_from_roi(roi_img, ocr, conf_min=CONF_MIN)
                        if out is not None:
                            num, conf = out
                            add_vote(track_id, num, conf)
                            got = True
                            # optional: Raw-Overlay oben an der Player-Box
                            if DRAW_ROI:
                                cv2.putText(frame, f"raw:{num} {conf:.2f}",
                                            (x1, max(0, y1-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                            print(f"[F{frame_idx} T{track_id} ROI=({rx1},{ry1},{rx2},{ry2})] #{num} conf={conf:.2f}")
  
                # OCR wurde (bei leg_ok) versucht -> Taktung/Assign updaten
                if leg_ok:
                    last_ocr_frame[track_id] = frame_idx
                    maybe_assign_number(track_id)
                frame_ocr_time += ti.time() - t2
            assigned = track_assigned_num.get(track_id, None)


            # Zeichnen
            label_txt = f"P-{dom_team} ID{track_id}"
            if assigned is not None:
                label_txt += f" #{assigned}"
            cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
            cv2.putText(frame, label_txt, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

            # Output puffern (conf aus Tracker unbekannt -> -1.0), extra = dom_team
            jersey_out = assigned if assigned is not None else -1
            output_rows.append((frame_idx, track_id, float(x1), float(y1), float(box_w), float(box_h), -1.0, dom_team, jersey_out))
    ocr_time += frame_ocr_time
    frame_time = ti.time() - frame_time
    print(f"Frame {frame_idx} Zeit: {frame_time*1000:.1f}ms "
      f"(Det: {frame_detection_time*1000:.1f}ms, "
      f"Track: {frame_track_time*1000:.1f}ms, "
      f"OCR: {frame_ocr_time*1000:.1f}ms)")

    # Frame schreiben
    video_writer.write(frame)

# Cleanup
video_writer.release()
cap.release()


# === Data-Postprocess ===
# --- 1)Rückwirkendes Auffülllen der Rücken-Nummern
final_numbers = {}
for tid in set([row[1] for row in output_rows]):
    if vote_sum_offline[tid]:
        best_num, _w = max(vote_sum_offline[tid].items(), key=lambda kv: kv[1])
        final_numbers[tid] = int(best_num)

filled_rows = []
for row in output_rows:
    frame_i, tid, x, y, w, h, conf_o, extra, jersey = row
    if extra in ('L','R') and tid in final_numbers:
        jersey = int(final_numbers[tid])
    filled_rows.append((frame_i, tid, x, y, w, h, conf_o, extra, jersey))

# --- 2) Majority-Vote Team pro Player-Track (extra in {'L','R'}) ---
team_votes = defaultdict(list)   # tid -> ['L','L','R',...]
for frame_i, tid, x, y, w, h, conf_o, extra, jersey in filled_rows:
    if extra in ('L', 'R'):
        team_votes[tid].append(extra)

team_final = {}
for tid, votes in team_votes.items():
    if votes:
        team_final[tid] = Counter(votes).most_common(1)[0][0]

# --- 3) Jersey pro Track final (sollte nach obigem Füllen konstant sein) ---
jersey_final = {}
for frame_i, tid, x, y, w, h, conf_o, extra, jersey in filled_rows:
    if extra in ('L','R') and jersey is not None and jersey != -1:
        jersey_final[tid] = int(jersey)

# --- 4) Key = (Team, Jersey) pro Track bilden (nur Spieler mit valider Jersey) ---
tid_key = {}
for tid in set([r[1] for r in filled_rows]):
    t = team_final.get(tid, None)
    j = jersey_final.get(tid, None)
    if t in ('L','R') and j is not None and 1 <= j <= 99:
        tid_key[tid] = (t, j)

# --- 5) Tracklets je (Team,Jersey) stitchen, wenn sie sich NICHT überlappen

# Spannen & erste/letzte Box pro tid
tid_span = {}        # tid -> [start_frame, end_frame]
tid_first = {}       # tid -> (frame, (x,y,w,h))
tid_last  = {}       # tid -> (frame, (x,y,w,h))

for frame_i, tid, x, y, w, h, conf_o, extra, jersey in filled_rows:
    if extra in ('L','R'):
        if tid not in tid_span:
            tid_span[tid] = [frame_i, frame_i]
            tid_first[tid] = (frame_i, (x, y, w, h))
        else:
            tid_span[tid][1] = frame_i
        tid_last[tid] = (frame_i, (x, y, w, h))

# (Team,Jersey) -> Liste von tids
key_to_tids = defaultdict(list)
for tid in set([r[1] for r in filled_rows]):
    t = team_final.get(tid, None)
    j = jersey_final.get(tid, None)
    if t in ('L','R') and j is not None and 1 <= j <= 99:
        key_to_tids[(t, j)].append(tid)

# Für jede (Team,Jersey)-Gruppe: Ketten ohne Überlappung bilden
tid_new_id = {}   # Mapping alt->neu

for key, tids in key_to_tids.items():
    # nur tids berücksichtigen, die Spannen haben
    tids = [t for t in tids if t in tid_span and t in tid_first and t in tid_last]
    if not tids:
        continue

    # nach Startframe sortieren
    tids_sorted = sorted(tids, key=lambda t: tid_span[t][0])

    chains = []  # Liste von Ketten; jede Kette ist eine Liste von tids

    for t in tids_sorted:
        placed = False
        t_start, t_end = tid_span[t]
        t_first_box = tid_first[t][1]
        # versuche, an bestehende Kette hinten anzudocken
        for chain in chains:
            prev = chain[-1]
            p_start, p_end = tid_span[prev]
            # keine zeitliche Überlappung: p_end < t_start
            if p_end < t_start:
                chain.append(t)
                placed = True
                break
        if not placed:
            chains.append([t])

    # IDs vergeben: jede Kette wird zu EINER Identität
    for chain in chains:
        new_id = min(chain)

        for t in chain:
            tid_new_id[t] = new_id

# --- 6) Relabeled Rows erzeugen:
#         - Für Spieler: Team via Majority-Vote ersetzen,
#           ID ggf. auf kanonische ID mappen,
#           Jersey bleibt wie zuvor gefüllt.
#         - Für GK/Ball/Ref bleibt alles wie geschrieben (du hast sie bereits ID=1 gesetzt).
relabeled_rows = []
for frame_i, tid, x, y, w, h, conf_o, extra, jersey in filled_rows:
    if extra in ('L', 'R'):  # Player
        team_out = team_final.get(tid, extra)  # vereinheitlicht
        new_id = tid_new_id.get(tid, tid)      # gemergte ID wenn Key vorhanden, sonst alte ID
        relabeled_rows.append((frame_i, new_id, x, y, w, h, conf_o, team_out, jersey))
    else:
        # GK (extra='GK-L'/'GK-R'), Ball (extra='2'), Ref (extra='3') – unverändert
        relabeled_rows.append((frame_i, tid, x, y, w, h, conf_o, extra, jersey))

# === Datei schreiben ===
with open(mot_file_path, 'w') as mot_writer:
    # Optional Header:
    # mot_writer.write("# frame, id, x, y, w, h, conf, extra(team/cid/side), jersey\n")
    for r in relabeled_rows:
        frame_i, tid, x, y, w, h, conf_o, extra, jersey = r
        # 'extra' ist Team ('L'/'R'), '1'/'3'/'2' für GK/Ref/Ball etc.
        mot_writer.write(f"{frame_i},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf_o:.2f},{extra},{jersey}\n")

total_time = ti.time() - start_time
other_time = total_time - (detection_time + track_time + ocr_time)
avg_det_ms  = (detection_time / max(1, frame_idx)) * 1000
avg_trk_ms  = (track_time     / max(1, frame_idx)) * 1000
avg_ocr_ms  = (ocr_time       / max(1, frame_idx)) * 1000
avg_tot_ms  = (total_time     / max(1, frame_idx)) * 1000
fps_eff     = (frame_idx / total_time) if total_time > 0 else 0.0

print("\n=== Runtime Summary ===")
print(f"Frames: {frame_idx} | Duration: {total_time:.2f}s | FPS: {fps_eff:.2f}")
print(f"Avg per frame [ms]  ->  Det: {avg_det_ms:.2f} | Track: {avg_trk_ms:.2f} | OCR: {avg_ocr_ms:.2f} | Total: {avg_tot_ms:.2f}")
print(f"Time share [%]       ->  Det: {100*detection_time/total_time:.1f} | Track: {100*track_time/total_time:.1f} | OCR: {100*ocr_time/total_time:.1f} | Other: {100*other_time/total_time:.1f}")
print(f"OCR calls: players={ocr_calls}, roi_calls={ocr_roi_calls}")
print(f"Video: {output_video}")
print(f"MOT-ähnlicher Output mit Jersey-Spalte: {mot_file_path}")