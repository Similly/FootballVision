import os
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort       
from sklearn.cluster import KMeans

# === Parameter ===
FRAME_DIR       = './data/tracking/train/SNMOT-060/img1'        # Ordner mit den Einzelbildern
OUTPUT_VIDEO    = 'output_cluster+track_3Teams.mp4'    # Ausgabedatei Video
FPS             = 25              # Bilder pro Sekunde
NUM_TEAMS       = 3
HIST_BINS       = [8, 8, 8]

# YOLO und SORT initialisieren
model   = YOLO('yolov8n.pt')
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# Dictionaries für Team-Zuordnung und Histogramme
track_team = {}
track_hist = {}

# Helfer: HSV-Histogramm für eine Bounding Box
def compute_hist(frame, box):
    x1, y1, x2, y2 = map(int, box)
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return np.zeros(np.prod(HIST_BINS))
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, HIST_BINS,
                        [0,180, 0,256, 0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# VideoWriter vorbereiten (Größe aus erstem Frame bestimmen)
first_frame_path = sorted(os.listdir(FRAME_DIR))[0]
first_frame = cv2.imread(os.path.join(FRAME_DIR, first_frame_path))
h, w = first_frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out   = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (w, h))

# Alle Frame-Dateien sortiert einlesen
time_step = 0
for fname in sorted(os.listdir(FRAME_DIR)):
    frame_path = os.path.join(FRAME_DIR, fname)
    frame = cv2.imread(frame_path)
    if frame is None:
        continue

    # Erkennung: Personen (class 0)
    results = model(frame, classes=[0])
    dets = []
    for r in results:
        for box in r.boxes.xyxy.cpu().numpy():
            dets.append(box)
    dets = np.array(dets) if dets else np.empty((0,4))

    # Tracking
    tracks = tracker.update(dets)

    # Team-Zuordnung & Zeichnung
    for t in tracks:
        x1, y1, x2, y2, track_id = map(int, t)
        if track_id not in track_team:
            # Neuer Track -> Histogramm berechnen
            hist = compute_hist(frame, (x1, y1, x2, y2))
            track_hist[track_id] = hist
            # Falls genug Tracks, k-means-Clustering aller gesammelten Histos
            if len(track_hist) >= NUM_TEAMS:
                hists = np.stack(list(track_hist.values()))
                labels = KMeans(n_clusters=NUM_TEAMS).fit_predict(hists)
                for tid, lbl in zip(track_hist.keys(), labels):
                    track_team[tid] = int(lbl)
            else:
                track_team[track_id] = 0

        team = track_team.get(track_id, 0)
        if team == 0:
            color = (0, 255, 0)
        elif team == 1:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {track_id} T{team}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Schreibe in Video
    out.write(frame)
    time_step += 1

# Freigeben
out.release()
print(f"Fertig! Video gespeichert unter: {OUTPUT_VIDEO}")