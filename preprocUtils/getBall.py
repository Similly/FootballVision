#!/usr/bin/env python3
import os
import glob

# Basisordner mit deinen Clips
BASE_DIR = './data/tracking/challenge'

for clip in os.listdir(BASE_DIR):
    labels_dir = os.path.join(BASE_DIR, clip, 'labels')
    if not os.path.isdir(labels_dir):
        continue

    # Alle Label-Dateien (*.txt) im labels-Ordner
    label_files = sorted(glob.glob(os.path.join(labels_dir, '*.txt')))
    if not label_files:
        continue

    print(f"Processing clip {clip}, {len(label_files)} frames…")
    for lf in label_files:
        # Jede Zeile: class_id x_center y_center width height
        with open(lf, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            continue

        # Parsen und Flächen berechnen (normalized area = w*h)
        entries = []
        for ln in lines:
            parts = ln.split()
            cls, xc, yc, w, h = parts
            area = float(w) * float(h)
            entries.append({
                'parts': parts,
                'area': area
            })

        # Index des kleinsten Objekts finden
        min_idx = min(range(len(entries)), key=lambda i: entries[i]['area'])

        # Class-ID auf "32" setzen
        entries[min_idx]['parts'][0] = '1'

        # Zurückschreiben (overschreibt original .txt)
        with open(lf, 'w') as f:
            for e in entries:
                f.write(' '.join(e['parts']) + '\n')

    print(f" → Done clip {clip}")
