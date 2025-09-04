#!/usr/bin/env python3
import os
import glob

# Basis-Pfad zu deinen Trainingssequenzen
BASE_DIR = './data/tracking/train_20pct'

# Muster für alle label-Dateien in jedem img1-Ordner
pattern = os.path.join(BASE_DIR, 'SNMOT-*', 'img1', '*.txt')

for lbl_path in glob.glob(pattern):
    # Datei einlesen
    with open(lbl_path, 'r') as f:
        lines = f.readlines()

    # Neu schreiben, Klasse 32 → 1 ersetzen
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        cls = parts[0]
        if cls == '32':
            parts[0] = '1'
        new_lines.append(' '.join(parts) + '\n')

    # Überschreibe die Datei nur, wenn sich was geändert hat
    if new_lines != lines:
        with open(lbl_path, 'w') as f:
            f.writelines(new_lines)
        print(f"Updated classes in {lbl_path}")

print("Fertig: alle Klasse-32-Labels wurden auf 1 umgeschrieben.")

