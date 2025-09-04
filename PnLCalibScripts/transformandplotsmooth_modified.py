import os
import cv2
import numpy as np
import pandas as pd
import argparse
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mplsoccer import Pitch
from OneEuroFilter import OneEuroFilter
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(
        description="Wendet Homographie-Transform-Matrizen auf Tracking-Daten an und erstellt ein Video mit overlaid Pitch-Plot und speichert 2D-Positionen.")
    parser.add_argument('--video',          required=True, help='Pfad zum Originalvideo (z.B. video.mp4)')
    parser.add_argument('--tracking',       required=True, help='Pfad zur MOT-Trackingdatei (frame,id,x,y,w,h,conf,class,jersey)')
    parser.add_argument('--transforms_csv', required=True, help='Pfad zur CSV mit den Homographien (frame,H00..H22)')
    parser.add_argument('--output_video',   required=True, help='Pfad für das Ausgabevideo (z.B. output_with_pitch.mp4)')
    parser.add_argument('--output_csv',     required=True, help='Pfad für die Ausgabedatei der 2D-Positionen (CSV)')

    # --- NEU: Parameter für Filter-Reset/Pruning ---
    parser.add_argument('--gap_seconds', type=float, default=2.0,
                        help='Reset, wenn Tracklet >= gap_seconds nicht gesehen wurde.')
    parser.add_argument('--jump_reset_thresh', type=float, default=8.0,
                        help='Reset bei Positionssprung (in Pitch-Metern) >= jump_reset_thresh.')
    parser.add_argument('--prune_after_seconds', type=float, default=15.0,
                        help='Tracklets aus Speicher löschen, wenn seit so vielen Sekunden nicht gesehen.')

    args = parser.parse_args()

    # Ausgabepfade anlegen
    os.makedirs(os.path.dirname(args.output_video), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv),   exist_ok=True)

    # Tracking-Daten laden (NEUES FORMAT)
    # class als str wegen 'L'/'R' und jersey als Int64 (nullable), -1 als "kein Wert"
    df_track = pd.read_csv(
        args.tracking,
        header=None,
        names=['frame','id','x','y','w','h','conf','class','jersey'],
        dtype={'frame': int, 'id': int, 'x': float, 'y': float, 'w': float, 'h': float,
               'conf': float, 'class': str, 'jersey': float}
    )
    # jersey zu int konvertieren, fehlende/-1 als -1
    df_track['jersey'] = df_track['jersey'].fillna(-1).astype(int)

    # Homographie-CSV laden
    df_h = pd.read_csv(args.transforms_csv, index_col='frame')
    # Erwartete Spalten: H00,H01,H02,H10,H11,H12,H20,H21,H22

    # Video einlesen
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise IOError(f"Kann Video nicht öffnen: {args.video}")
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    ori_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ori_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Ausgabegröße (anpassen, falls gewünscht)
    out_w, out_h = 960, 540
    sx, sy = out_w / max(1, ori_w), out_h / max(1, ori_h)

    # Filter + Re-Entry-State
    filters   = {}      # (class,id) -> {'x': OneEuroFilter, 'y': OneEuroFilter}
    last_seen = {}      # (class,id) -> letzter frame_id (int)
    last_pos  = {}      # (class,id) -> (mx, my) in Pitchkoordinaten (zur Jump-Erkennung)

    # --- Helper für Reset-Logik ---
    def make_filters_for_key():
        return {
            'x': OneEuroFilter(freq=fps, mincutoff=0.8, beta=0.004),
            'y': OneEuroFilter(freq=fps, mincutoff=0.8, beta=0.004),
        }

    def need_gap_reset(key, frame_id):
        """Reset, wenn Tracklet lange nicht gesehen wurde oder neu."""
        if key not in last_seen:
            return True  # erster Frame dieses Keys -> init
        gap_frames = frame_id - last_seen[key]
        return gap_frames >= int(round(args.gap_seconds * fps))

    def need_jump_reset(key, mx, my):
        """Reset bei großem Sprung in Pitch-Metern (z.B. Re-ID)."""
        if key not in last_pos:
            return False
        px, py = last_pos[key]
        dist = ((mx - px)**2 + (my - py)**2) ** 0.5
        return dist >= args.jump_reset_thresh

    # VideoWriter (H.264; falls nicht unterstützt, auf 'mp4v' wechseln)
    fourcc    = cv2.VideoWriter_fourcc(*'avc1')
    out_video = cv2.VideoWriter(args.output_video, fourcc, fps, (out_w, out_h))
    if not out_video.isOpened():
        fourcc    = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(args.output_video, fourcc, fps, (out_w, out_h))

    # Output-CSV öffnen (JETZT MIT jersey)
    csv_file = open(args.output_csv, 'w')
    csv_file.write('frame,id,model_x,model_y,class,jersey\n')

    # Pitch-Plot vorbereiten
    dpi = 100
    template_w, template_h = 105, 68
    plot_w_px, plot_h_px = int(template_w * 1.3), int(template_h * 1.3)

    fig = plt.figure(figsize=(plot_w_px / dpi, plot_h_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    pitch = Pitch(
        pitch_type='custom',
        pitch_length=template_w, pitch_width=template_h,
        line_color='white', pitch_color='grass', stripe=True, linewidth=1,
    )
    pitch.draw(ax=ax, figsize=(2, 1.3), tight_layout=True)
    fig.patch.set_alpha(0)
    ax.axis('off')

    canvas = FigureCanvas(fig)
    fig.canvas.draw()
    plot_w, plot_h = fig.canvas.get_width_height()

    # zentrieren
    offset_x = (out_w - plot_w) // 2
    offset_y = 0

    # Farben (BGR für OpenCV)
    bgr_colors = {
        'L': (0, 0, 255),    # rot
        'R': (0, 255, 0),    # grün
        '2': (255, 255, 255),# weiß (Ball)
        '3': (0, 255, 255),  # gelb (Ref)
    }
    # Farben (RGB 0..1 für Matplotlib)
    rgb_colors = {
        'L': (1.0, 0.0, 0.0),   # rot
        'R': (0.0, 1.0, 0.0),   # grün
        '2': (1.0, 1.0, 1.0),   # weiß
        '3': (1.0, 1.0, 0.0),   # gelb
    }

    prev_H = None
    T_accept = 1.5
    T_reject = 6.0
    unstable = False
    unstable_count = 0
    MAX_UNSTABLE = 5

    frame_id = 0
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (out_w, out_h))

        # Tracking dieses Frames (Achtung: viele Tracker loggen 1-basiert)
        rows = df_track[df_track['frame'] == frame_id + 1]  # ggf. +0, falls 0-basiert

        # Homographie aus CSV holen
        if frame_id not in df_h.index:
            H = np.eye(3, dtype=np.float32)
        else:
            row = df_h.loc[frame_id, ['H00','H01','H02','H10','H11','H12','H20','H21','H22']].values
            H = row.reshape(3, 3).astype(np.float32)

            # Stabilitätscheck nur, wenn Detections existieren
            if prev_H is not None and not rows.empty:
                # Bildpunkte der aktuellen Detections sammeln
                img_pts = []
                for _, det in rows.iterrows():
                    cx = (det['x'] + det['w'] / 2.0) * sx
                    cy = (det['y'] + det['h'])       * sy
                    img_pts.append([cx, cy])

                if len(img_pts) > 0:
                    img_pts = np.array(img_pts, dtype=np.float32).reshape(-1, 1, 2)
                    old_model = cv2.perspectiveTransform(img_pts, np.linalg.inv(prev_H))
                    new_model = cv2.perspectiveTransform(img_pts, np.linalg.inv(H))
                    diffs = np.linalg.norm((new_model - old_model).reshape(-1, 2), axis=1)
                    med = float(np.median(diffs))

                    if not unstable:
                        if med > T_accept:
                            unstable = True
                            unstable_count += 1
                            H = prev_H.copy()
                        else:
                            prev_H = H.copy()
                    else:
                        if med < T_reject:
                            unstable = False
                            unstable_count = 0
                            prev_H = H.copy()
                        else:
                            if unstable_count >= MAX_UNSTABLE:
                                prev_H = H.copy()
                                unstable = False
                                unstable_count = 0
                            else:
                                unstable_count += 1
                                H = prev_H.copy()
            else:
                prev_H = H.copy()

        # Inverse H robust berechnen
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            H_inv = np.linalg.inv(prev_H) if prev_H is not None else np.eye(3, dtype=np.float32)

        # Positionen in Modellkoordinaten + Zeichnen
        positions = []
        for _, det in rows.iterrows():
            key = (str(det['class']), int(det['id']))
            if key not in filters:
                filters[key] = make_filters_for_key()

            # Fußpunkt berechnen (Bild -> Pitch)
            cx = (det['x'] + det['w']/2.0) * sx
            cy = (det['y'] + det['h'])     * sy
            pt = np.array([[[cx, cy]]], dtype=np.float32)
            mx_rel, my_rel = cv2.perspectiveTransform(pt, H_inv)[0, 0]

            # Welt-Ursprung war in der Mitte
            mx = mx_rel + template_w / 2.0
            my = -my_rel + template_h / 2.0

            # --- NEU: Reset-Logik gegen "Reinfliegen" ---
            gap_reset  = need_gap_reset(key, frame_id)
            jump_reset = need_jump_reset(key, mx, my)

            if gap_reset or jump_reset:
                # Filter neu initialisieren
                filters[key] = make_filters_for_key()

                # No-fly-in: im ersten Frame nach Reset direkt Rohwert ausgeben
                mx_smooth = mx
                my_smooth = my

                # Filter mit aktuellem Wert primen (für nächsten Frame)
                _ = filters[key]['x'](mx)
                _ = filters[key]['y'](my)
            else:
                # Normales Glätten
                mx_smooth = filters[key]['x'](mx)
                my_smooth = filters[key]['y'](my)

            # State updaten
            last_seen[key] = frame_id
            last_pos[key]  = (mx, my)

            cls = str(det['class'])
            jersey = int(det['jersey']) if pd.notna(det['jersey']) else -1

            # CSV-Zeile (JETZT MIT jersey)
            csv_file.write(f"{frame_id},{int(det['id'])},{mx_smooth:.2f},{my_smooth:.2f},{cls},{jersey}\n")
            positions.append((mx_smooth, my_smooth, cls))

            # Bounding-Box auf Video zeichnen
            x1, y1 = int(det['x'] * sx), int(det['y'] * sy)
            w_, h_  = int(det['w'] * sx), int(det['h'] * sy)

            color_bgr = bgr_colors.get(cls, (0, 0, 0))
            cv2.rectangle(frame, (x1, y1), (x1 + w_, y1 + h_), color_bgr, 2)

            jersey_txt = f" #{jersey}" if jersey >= 0 else " #?"
            label_txt  = f"{cls}-{int(det['id'])}{jersey_txt}"
            cv2.putText(frame, label_txt, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1, lineType=cv2.LINE_AA)

        # --- NEU: Pruning alter Keys (Speicher sauber halten) ---
        prune_frames = int(round(args.prune_after_seconds * fps))
        if prune_frames > 0:
            to_del = []
            for k, last_f in list(last_seen.items()):
                if frame_id - last_f >= prune_frames:
                    to_del.append(k)
            for k in to_del:
                last_seen.pop(k, None)
                filters.pop(k, None)
                last_pos.pop(k, None)

        # Pitch neu zeichnen (einfach gehalten)
        ax.clear()
        pitch.draw(ax=ax)

        # Punkte nach Klasse gruppieren und plotten (RGB)
        groups = defaultdict(list)
        for x, y, c in positions:
            groups[c].append((x, y))

        for c, pts in groups.items():
            if not pts:
                continue
            xs, ys = zip(*pts)
            color_rgb = rgb_colors.get(c, (0.0, 0.0, 0.0))
            ax.scatter(xs, ys, s=10, c=[color_rgb], edgecolors='black', linewidths=0.5)

        # Rendern & Overlay
        canvas.draw()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(plot_h, plot_w, 4)
        rgb   = buf[..., :3]
        alpha = buf[..., 3:4].astype(float) / 255.0

        overlay = rgb.astype(float)
        base    = frame[offset_y:offset_y+plot_h, offset_x:offset_x+plot_w].astype(float)
        composited = (alpha * overlay + (1 - alpha) * base).astype(np.uint8)
        frame[offset_y:offset_y+plot_h, offset_x:offset_x+plot_w] = composited

        cv2.putText(frame, f"{frame_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out_video.write(frame)
        print(f"Frame {frame_id} in {time.time() - start:.3f}s")
        frame_id += 1

    cap.release()
    out_video.release()
    csv_file.close()
    plt.close(fig)

if __name__ == '__main__':
    main()
