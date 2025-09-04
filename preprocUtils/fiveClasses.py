import os
import configparser
from pathlib import Path
from collections import defaultdict
from PIL import Image

# Root-Ordner, in dem sich pro Sequenz ein Unterordner SNMOT-*/ mit gameinfo.ini, gt.txt und img1/ befindet
SEQUENCES_ROOT = Path('../data/tracking/test')  # <–– hier anpassen
#OUTPUT_ROOT   = Path('labels5')

# Mapping Klassenname → YOLO-Index
CLASS_MAP = {
    'player':           0,
    'goalkeeper':       1,
    'ball':             2,
    'main referee':     3,
    'side referee':     4,
    'other':            5,
}

def parse_gameinfo(path_ini: Path):
    """Liest gameinfo.ini ein und gibt dict TrackletID(int)→Klassenindex."""
    print(f"    -> Lese Klassen aus {path_ini}")
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(path_ini, encoding='utf-8')
    mapping = {}
    for key, val in cfg['Sequence'].items():
        if not key.startswith('trackletID_'):
            continue
        tid = int(key.split('_', 1)[1])
        parts = [p.strip().lower() for p in val.split(';')]
        raw_cls = parts[0]      # z.B. 'player team right', 'referee', 'goalkeeper team right', 'other', 'ball'
        qual    = parts[1] if len(parts) > 1 else ''

        # Generalisiere auf eine der sechs Gruppen
        if raw_cls.startswith('player'):
            cls = CLASS_MAP['player']
        elif raw_cls.startswith('goalkeeper'):
            cls = CLASS_MAP['goalkeeper']
        elif raw_cls == 'ball':
            cls = CLASS_MAP['ball']
        elif raw_cls == 'referee':
            # Unterscheide Haupt- vs. Seiten-Schiedsrichter
            if 'main' in qual:
                cls = CLASS_MAP['main referee']
            else:
                cls = CLASS_MAP['side referee']
        elif raw_cls == 'other':
            cls = CLASS_MAP['other']
        else:
            # Fallback, falls ein unerwarteter Wert auftaucht
            raise ValueError(f"Unbekannte Klasse '{raw_cls}' in {path_ini}")

        mapping[tid] = cls

    print(f"      -> {len(mapping)} Tracklets gemappt")
    return mapping

def parse_gt(path_gt: Path):
    """
    Liest gt.txt ein und gibt dict frame→Liste von (trackletID,x,y,w,h).
    Annahme: Spalte-2=trackletID, Spalte-3/4/5/6 = x,y,w,h
    """
    print(f"    -> Lese GT-Bounding-Boxes aus {path_gt}")
    frames = defaultdict(list)
    with open(path_gt, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6: 
                continue
            # parts: [bboxID, trackletID, x, y, w, h, ...]
            tid = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            frame = int(parts[0])  # falls Spalte 1 Frame ist, sonst anpassen: frame=int(parts[?])
            # Wenn stattdessen parts[1] der Frame ist, frame=int(parts[1]) und tid=int(parts[0])!
            frames[frame].append((tid, x, y, w, h))
    
    if frames:
        print(f"      -> Gefundene Frames: {len(frames)}, erste: {min(frames)}")
    else:
        print("      -> Keine Frames gefunden!")
    return frames

def get_image_size(img_dir: Path, frame: int):
    """
    Holt Breite/Höhe aus dem ersten Bild in img_dir passend zum Frame.
    Erwartet Dateinamen wie 000001.jpg oder ähnlich.
    """
    # hier: zero-padded 6-stellig
    fname = img_dir / f"{frame:06d}.jpg"
    if not fname.exists():
        # alternativ alle Bilder durchsuchen
        candidates = sorted(img_dir.glob(f"*{frame:06d}*"))
        if candidates:
            fname = candidates[0]
        else:
            raise FileNotFoundError(f"Frame {frame} in {img_dir} nicht gefunden.")
    with Image.open(fname) as im:
        return im.width, im.height

def write_yolo_labels(seqr: Path):
    print(f"Verarbeite Sequenz: {seqr.name}")
    ini  = seqr / 'gameinfo.ini'
    gt   = seqr / 'gt/gt.txt'
    img1 = seqr / 'img1'
    if not ini.exists() or not gt.exists() or not img1.is_dir():
        print(f"  skippe {seqr.name}, fehlt ini/gt/img1")
        return

    tracklet2cls = parse_gameinfo(ini)
    frames = parse_gt(gt)

    out_dir = seqr/'labels5'
    out_dir.mkdir(exist_ok=True)

    total = 0
    for frame, boxes in frames.items():
        w_img, h_img = get_image_size(img1, frame)
        lines = []
        for tid, x, y, w, h in boxes:
            cls = tracklet2cls.get(tid, None)
            if cls is None:
                continue  # unbekannte TrackletID
            # YOLO verlangt x_center,y_center
            x_c = x + w/2
            y_c = y + h/2
            # normalisieren
            x_c /= w_img
            y_c /= h_img
            w   /= w_img
            h   /= h_img
            lines.append(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

        # schreibe Datei nur, wenn es Boxen gibt
        if lines:
            fn = out_dir / f"{frame:06d}.txt"
            with open(fn, 'w') as f:
                f.write("\n".join(lines))
            total += 1
    print(f"  -> Fertig: {total} Label-Dateien geschrieben\n")

if __name__ == '__main__':
    print("Suche Sequenzen unter:", SEQUENCES_ROOT)
    print("Existiert?", SEQUENCES_ROOT.exists())
    seqs = sorted(SEQUENCES_ROOT.glob('SNMOT-*'))
    print("Gefundene Sequenzen:", [s.name for s in seqs])
    if not seqs:
        print(f"Keine Sequenzen gefunden unter {SEQUENCES_ROOT}")
        #sys.exit(1)
    for seq in seqs:
        print(f"Verarbeite Sequenz {seq.name}")
        write_yolo_labels(seq)
    print("=== Fertig alle Sequenzen ===") 
