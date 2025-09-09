import cv2
import re
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

from .config import MIN_PLAYER_H, USE_PARSEQ

# === PARSeq: transform factory ===
def make_parseq_transform(img_size, rotation=0, augment=False):
    trans = []
    if rotation:
        trans.append(lambda img: img.rotate(rotation, expand=True))
    trans.extend([
        T.Resize(img_size, T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return T.Compose(trans)

def preprocess_roi_parseq(roi_bgr):
    H, W = roi_bgr.shape[:2]
    scale = 2 if max(H, W) < 200 else 1
    if scale != 1:
        roi_bgr = cv2.resize(roi_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return rgb

def jersey_rois_with_abs(frame, x1, y1, x2, y2):
    H = max(0, y2 - y1)
    W = max(0, x2 - x1)
    if H < MIN_PLAYER_H or W < 40:
        return []

    # back torso ROI
    ry1, ry2 = int(y1 + 0.15 * H), int(y1 + 0.55 * H)
    rx1, rx2 = int(x1 + 0.15 * W), int(x1 + 0.85 * W)
    back_box = (rx1, ry1, rx2, ry2)

    hF, wF = frame.shape[:2]
    def clamp_box(b):
        x1b, y1b, x2b, y2b = b
        x1b = max(0, min(wF - 1, x1b)); x2b = max(0, min(wF - 1, x2b))
        y1b = max(0, min(hF - 1, y1b)); y2b = max(0, min(hF - 1, y2b))
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

def load_parseq_from_ckpt(ckpt_path, device):
    import string as _s
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
    fixed = {}
    for k, v in sd.items():
        k = k.replace("module.", "").replace("_orig_mod.", "")
        if not k.startswith("model."):
            k = "model." + k
        fixed[k] = v

    pq = fixed["model.pos_queries"]
    n_queries = pq.shape[1]
    d_model = pq.shape[2]
    arch = "parseq_tiny" if d_model <= 192 else "parseq"
    max_len = int(n_queries - 1)

    head_out = fixed["model.head.weight"].shape[0]
    embed_num = fixed["model.text_embed.embedding.weight"].shape[0]

    hp = ckpt.get("hyper_parameters", {})
    if head_out == 11:
        charset_str = _s.digits
    else:
        charset_str = hp.get("charset") or "".join([c for c in _s.printable if c not in "\t\n\r\x0b\x0c"])

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
        model = _build()
        model.model.head = torch.nn.Linear(d_model, head_out, bias=True)
        model.model.text_embed.embedding = torch.nn.Embedding(embed_num, d_model)
        try:
            if head_out == 11:
                model.tokenizer.set_charset(charset_str)
        except Exception:
            try:
                model.tokenizer.charset = charset_str
            except Exception:
                pass

    missing, unexpected = model.load_state_dict(fixed, strict=False)
    if missing or unexpected:
        print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")

    model = model.to(device).eval()
    tfm = make_parseq_transform(tuple(hp.get("img_size", (32, 128))))
    print(f"[PARSeq] arch={arch} d_model={d_model} max_len={max_len} head_out={head_out} embed_num={embed_num}")
    return model, tfm

def parseq_infer_text(bgr_roi, model, tf, device):
    if bgr_roi is None or bgr_roi.size == 0:
        return "", 0.0
    rgb = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2RGB)
    x = tf(Image.fromarray(rgb)).unsqueeze(0).to(device)  # [1,3,H,W]

    logits = model(x)                  # [B, T, C]
    logits = logits[:, :3, :11]        # 3 positions, 10 digits + <eos>
    probs = F.softmax(logits, dim=-1)  # [B, 3, 11]

    preds, probs_list = model.tokenizer.decode(probs)
    text = preds[0]
    if isinstance(probs_list, (list, tuple)) and len(probs_list) > 0:
        conf = float(torch.as_tensor(probs_list[0]).prod().item())
    else:
        conf = float(probs_list)
    return text, conf

def read_number_from_roi(roi_bgr, ocr_model, tf, device, conf_min=0.50):
    H, W = roi_bgr.shape[:2]
    ar = W / max(1, H)
    if not (0.2 <= ar <= 5.0):
        return None

    if USE_PARSEQ:
        pre_img = preprocess_roi_parseq(roi_bgr)
    else:
        # (kept for parity; currently unused when USE_PARSEQ=True)
        pre_img = roi_bgr

    texts, scores, boxes, polys = [], [], None, None
    text, conf = parseq_infer_text(pre_img, ocr_model, tf, device)
    texts = [text]; scores = [conf]

    if len(texts) == 0:
        return None

    roi_cx, roi_cy = W * 0.5, H * 0.5

    best = None
    for i, (txt, rec_conf) in enumerate(zip(texts, scores)):
        digits = re.sub(r"[^0-9]", "", str(txt))
        if len(digits) == 0 or len(digits) > 2:
            continue
        val = int(digits)
        if not (1 <= val <= 99):
            continue

        # geometric prior (center preference)
        dist = 0.0  # no boxes/polys here, keep behavior: center of ROI
        geom = max(0.7, 1.0 - 0.5 * dist)
        c = float(rec_conf) * float(geom)

        if best is None or c > best[1]:
            best = (val, c)

    if best is None or best[1] < conf_min:
        return None
    return best
