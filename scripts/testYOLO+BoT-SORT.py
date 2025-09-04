import os
import sys
import cv2
from types import SimpleNamespace
from ultralytics import YOLO
import numpy as np
if not hasattr(np, 'float'):
    np.float = float
from tracker.bot_sort import BoTSORT
import time as ti

start_time = ti.time()


# === Parameter ===
image_dir = "data/tracking/train/SNMOT-062/img1"
output_video = "outputVids/track/output062_BoT-SORT_2.mp4"
fps = 25

# === Bilddateien laden ===
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))])
first_img = cv2.imread(os.path.join(image_dir, image_files[0]))
height, width = first_img.shape[:2]

# === YOLOv8 Modell laden ===
model = YOLO("detModels/yolov8n+SoccerNet5class_phase2/weights/best.pt")

# === ByteTrack-Tracker initialisieren ===
args = SimpleNamespace(
    track_high_thresh=0.5,         # Mindestscore für high confidence Detections           # Mindestscore für Detections, die verfolgt werden sollen
    track_low_thresh=0.1,          # Untergrenze, ab der Detections fürs Verlängern genutzt werden
    new_track_thresh=0.7,          # Score für Start eines neuen Tracks
    track_buffer=25,               # 30 Frames (~1 Sekunde bei 30 FPS)
    mot20=False,                   # Kein spezieller MOT20-Modus
    proximity_thresh=0.4,          # IOU Threshold für Matching
    match_thresh=0.8,
    appearance_thresh=0.25,        # Feature-Distanz für ReID
    with_reid=False,                # Appearance Matching aktivieren
    fast_reid_config="reid/bagtricks_S50.yml",        # Pfad zur ReID-Konfiguration
    fast_reid_weights="reid/market_bot_S50.pth",   # Pretrained ReID Model
    device="cpu",                 # oder "cpu"
    cmc_method="sift",             # oder "ecc", falls du Probleme mit SIFT hast
    name="BoTSORT",                # Logging-Name
    ablation="",                   # Leerlassen (Standard)
)

# === VideoWriter vorbereiten ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

colors = {
    0: (255, 0, 0),    # Klasse 0: Blau - Player
    1: (0, 255, 0),    # Klasse 1: Grün - Keeper
    2: (0, 0, 255),    # Klasse 2: Rot - Ball
    3: (255, 255, 0),  # Klasse 3: Cyan - Main Ref
    4: (0, 255, 255),  # Klasse 4: Gelb - Side Ref
    5: (255, 0, 255),  # Klasse 5: Magenta - Other
}
target_classes = [0, 1, 2, 3, 4, 5]  # Klassen, die verfolgt werden sollen
trackers_dict = {}
for class_id in target_classes:
    trackers_dict[class_id] = BoTSORT(args)
    print(f"Tracker für Klasse {class_id} initialisiert.")

detection_time = 0
track_time = 0
for img_name in image_files:
    img_path = os.path.join(image_dir, img_name)
    frame = cv2.imread(img_path)

    # YOLOv8 Inferenz (alle Detections im Frame)
    start_time_det = ti.time()
    results = model(frame)
    end_time_det = ti.time() - start_time_det
    detection_time += end_time_det
    print(f"Inferenzzeit für {img_name}: {end_time_det:.2f} Sekunden")
          
    detections = results[0].boxes  # bboxes, scores, classes

    # Detektionen für ByteTrack aufbereiten: [x1, y1, x2, y2, score, class_id]
    dets = []
    for box in detections:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        score = float(box.conf[0])
        class_id = int(box.cls[0])
        # Beispiel: NUR Personen-Tracking (Class-ID anpassen falls nötig)
        #if class_id == 0:  # 0 = Person (in deinem Modell ggf. prüfen)
        dets.append([x1, y1, x2, y2, score, class_id])
        #print(f"Detektion: {x1}, {y1}, {x2}, {y2}, {score}, {class_id}")
    dets = np.array(dets)

    #print(dets.shape[1])

    # ByteTrack update
    # Format: dets muss shape [N, 6] sein: [x1, y1, x2, y2, score, class_id]
    #frame = np.asarray(frame, dtype=np.float32)
    #online_targets = tracker.update(dets, frame)
    start_time_track = ti.time()
    results_by_class = {}  # Optional: Ergebnisse pro Klasse speichern
    for class_id in target_classes:
        class_dets = dets[dets[:, 5] == class_id] if dets.size else np.empty((0, 6))[:, :5]
        print(f"Class {class_id} Detections: {class_dets.shape[0]}")
        if class_dets.shape[1] == 6:
            dets_for_track = class_dets[:, :5]
            print("Länge: 1")
        else:
            dets_for_track = class_dets
            print("Länge: 2")
        online_targets = trackers_dict[class_id].update(dets_for_track, frame)
        results_by_class[class_id] = online_targets
        print(f"Online Targets für Klasse {class_id}: {len(online_targets)}")   

        # Visualisierung: Bounding Boxes mit Tracking-IDs anzeigen
        color = colors.get(class_id, (255, 255, 255))  # Fallback: Weiß
        for t in online_targets:
            tlwh = t.tlwh  # [x, y, w, h]
            track_id = t.track_id
            x1, y1, w, h = [int(i) for i in tlwh]
            x2, y2 = x1 + w, y1 + h
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'Class: {class_id} ID: {track_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    end_time_track = ti.time() - start_time_track
    track_time += end_time_track
    print(f"Tracking-Zeit für {img_name}: {end_time_track:.2f} Sekunden")
    
    # Frame ins Video schreiben
    print(f"Frame: {img_name}")
    video_writer.write(frame)

video_writer.release()

end_time = ti.time()
print(f"Fertig! {len(image_files)} Bilder verarbeitet in {end_time - start_time:.2f} Sekunden. \nDetektion Zeit: {detection_time:.2f} Sekunden. \nTracking Zeit: {track_time:.2f} Sekunden.")
print(f"Video gespeichert unter: {output_video}")
