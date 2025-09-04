import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time as t

# Pfad zu deinen Einzelbildern
image_dir = "data/tracking/train/SNMOT-062/img1"  # Beispiel: "./match_frames"

output_video = "outputVids/det/output062_2.mp4"
fps = 25

# === Bilddateien laden ===
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))])
first_img = cv2.imread(os.path.join(image_dir, image_files[0]))
height, width = first_img.shape[:2]

# === YOLOv8 Modell laden ===
model = YOLO("detModels/yolov8n+SoccerNet5class_phase2/weights/best.pt")  # oder yolov8s.pt / yolov8m.pt

# === VideoWriter vorbereiten ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# === Alle Bilder analysieren & ins Video schreiben ===
start_time = t.time()
for fname in image_files:
    img_path = os.path.join(image_dir, fname)
    frame = cv2.imread(img_path)
    results = model(frame)[0]

    # definiere BGR-Farben für jede Klasse
    color_map = {
        'player':        (0, 255, 0),     # Grün
        'goalkeeper':    (255, 0, 0),     # Blau
        'ball':          (0, 165, 255),   # Orange
        'main referee':  (0, 0, 255),     # Rot
        'side referee':  (255, 255, 0),   # Cyan
        'other':         (128, 128, 128), # Grau
    }

    # Loop über alle erkannten Boxes
    for box in results.boxes:
        cls_id = int(box.cls)            # Klassen-ID (0–5)
        label  = model.names[cls_id]     # z.B. "player", "goalkeeper", …
        conf   = float(box.conf)         # Confidence-Wert

        # Bounding-Box-Koordinaten
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Farbe aus dem Mapping, default Weiß, falls Name nicht gefunden
        color = color_map.get(label, (255, 255, 255))

        # Zeichnen
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )

    print(f"Frame: {fname}")
    video_writer.write(frame)

end_time = t.time()
print(f"Fertig! {len(image_files)} Bilder verarbeitet in {end_time - start_time:.2f} Sekunden.")

video_writer.release()
print(f"Video gespeichert unter: {output_video}")