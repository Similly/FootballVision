import os
import cv2

# Basis-Pfad zu deinen Clips
BASE_DIR = './data/tracking/challenge'

for clip in os.listdir(BASE_DIR):
    clip_dir = os.path.join(BASE_DIR, clip)
    det_file = os.path.join(clip_dir, 'det', 'det.txt')
    img_dir  = os.path.join(clip_dir, 'img1')
    labels_dir = os.path.join(clip_dir, 'labels')

    # überspringen, falls kein det.txt oder kein Bildordner existiert
    if not os.path.isfile(det_file) or not os.path.isdir(img_dir):
        continue

    os.makedirs(labels_dir, exist_ok=True)
    # Bildgröße aus erstem Frame ermitteln
    first_img = sorted(os.listdir(img_dir))[0]
    img0 = cv2.imread(os.path.join(img_dir, first_img))
    h, w = img0.shape[:2]

    # det.txt im MOT20-Format: frame, id, x, y, w_box, h_box, ...
    with open(det_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            frame_idx = int(parts[0])
            # track_id = parts[1]    # falls du die ID brauchst
            x1 = float(parts[2])
            y1 = float(parts[3])
            bw = float(parts[4])
            bh = float(parts[5])

            # YOLO-Format: class x_center y_center width height (alle normiert)
            xc = (x1 + bw/2) / w
            yc = (y1 + bh/2) / h
            nw = bw / w
            nh = bh / h

            class_id = 0  # z.B. 0 = Person, passe an wenn du mehrere Klassen hast

            # Label-Datei pro Bild (000001.txt, 000002.txt, …)
            label_path = os.path.join(labels_dir, f"{frame_idx:06d}.txt")
            with open(label_path, 'a') as out:
                out.write(f"{class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

    print(f"Converted {det_file} → labels in {labels_dir}")