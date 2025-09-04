import os
import sys
import cv2
from ultralytics import YOLO
import numpy as np
if not hasattr(np, 'float'):
    np.float = float

bytetrack_path = os.path.abspath("tracker/ByteTrack")
if bytetrack_path not in sys.path:
    sys.path.insert(0, bytetrack_path)
from yolox.tracker.byte_tracker import BYTETracker  # ByteTrack-Tracker importieren

# === Parameter ===
image_dir = "data/tracking/train/SNMOT-061/img1"
output_video = "outputVids/track/output061_byteTrack_3.mp4"
fps = 25

class TrackerArgs:
    track_thresh = 0.5
    match_thresh = 0.8
    track_buffer = 60
    device = 1
    mot20 = False

args = TrackerArgs()

# === Bilddateien laden ===
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))])
first_img = cv2.imread(os.path.join(image_dir, image_files[0]))
height, width = first_img.shape[:2]

# === YOLOv8 Modell laden ===
args = TrackerArgs()
model = YOLO("detModels/yolov8n+SoccerNet5class_phase2/weights/best.pt")

# === ByteTrack-Tracker initialisieren ===
tracker = BYTETracker(args)  # mit Default-Parametern, anpassbar für bessere Performance

# === VideoWriter vorbereiten ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

for img_name in image_files:
    img_path = os.path.join(image_dir, img_name)
    frame = cv2.imread(img_path)

    # YOLOv8 Inferenz (alle Detections im Frame)
    results = model(frame)
    detections = results[0].boxes  # bboxes, scores, classes

    # Detektionen für ByteTrack aufbereiten: [x1, y1, x2, y2, score, class_id]
    dets = []
    for box in detections:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        score = float(box.conf[0])
        class_id = int(box.cls[0])
        # Beispiel: NUR Personen-Tracking (Class-ID anpassen falls nötig)
        if class_id == 0:  # 0 = Person (in deinem Modell ggf. prüfen)
            dets.append([x1, y1, x2, y2, score])
    dets = np.array(dets)

    print(dets.shape[1])

    # ByteTrack update
    # Format: dets muss shape [N, 6] sein: [x1, y1, x2, y2, score, class_id]
    online_targets = tracker.update(dets, (height, width), (height, width))

    # Visualisierung: Bounding Boxes mit Tracking-IDs anzeigen
    for t in online_targets:
        tlwh = t.tlwh  # [x, y, w, h]
        track_id = t.track_id
        x1, y1, w, h = [int(i) for i in tlwh]
        x2, y2 = x1 + w, y1 + h
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    # Frame ins Video schreiben
    video_writer.write(frame)

video_writer.release()
print(f"Video gespeichert unter: {output_video}")
