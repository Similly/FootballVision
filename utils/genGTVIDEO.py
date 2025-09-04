import os
import cv2
import glob

# === Konfiguration ===
BASE_DIR = './data/tracking/train'
CLIP     = 'SNMOT-100'          # hier auf deine Sequenz setzen
FPS      = 25
FOURCC   = cv2.VideoWriter_fourcc(*'mp4v')

# Pfade für die eine Sequenz
clip_dir = os.path.join(BASE_DIR, CLIP)
img_dir  = os.path.join(clip_dir, 'img1')
lbl_dir  = os.path.join(clip_dir, 'labels')
out_path = os.path.join(clip_dir, f'viz_{CLIP}.mp4')

# Prüfen, ob alles da ist
if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
    raise FileNotFoundError(f"Ordner nicht gefunden: {img_dir} oder {lbl_dir}")

# Bildliste
img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
if not img_files:
    raise FileNotFoundError(f"Keine Bilder in {img_dir}")

# VideoWriter initialisieren
frame0 = cv2.imread(img_files[0])
h, w = frame0.shape[:2]
writer = cv2.VideoWriter(out_path, FOURCC, FPS, (w, h))

# Frames durchgehen, Labels einzeichnen, Video schreiben
for img_path in img_files:
    fname  = os.path.basename(img_path)
    base   = os.path.splitext(fname)[0]
    lbl_fp = os.path.join(lbl_dir, f'{base}.txt')

    frame = cv2.imread(img_path)
    # Labels laden und zeichnen
    if os.path.isfile(lbl_fp):
        for line in open(lbl_fp, 'r'):
            cls, xc, yc, bw, bh = line.split()
            cls = int(cls)
            xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
            # Normalisiert → Pixel
            x1 = int((xc - bw/2) * w)
            y1 = int((yc - bh/2) * h)
            x2 = int((xc + bw/2) * w)
            y2 = int((yc + bh/2) * h)

            color = (255,0,0) if cls == 0 else (0,0,255)  # 0=Blau, 32=Rot
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, str(cls), (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    writer.write(frame)

writer.release()
print(f"[✓] Video mit GT-Boxes für {CLIP} gespeichert in:\n    {out_path}")
