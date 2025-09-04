import os
import numpy as np
import pandas as pd
import argparse
from time import perf_counter
import cv2  # nur für cv2.perspectiveTransform (keine Video-IO!)
from OneEuroFilter import OneEuroFilter

def main():
    parser = argparse.ArgumentParser(
        description="Transformiert MOT-Trackingdaten mit Homographien in Pitch-Koordinaten (ohne Video-Rendering)."
    )
    parser.add_argument('--tracking',       required=True, help='Pfad zur MOT-Trackingdatei (frame,id,x,y,w,h,conf,class,jersey)')
    parser.add_argument('--transforms_csv', required=True, help='Pfad zur CSV mit den Homographien (frame,H00..H22)')
    parser.add_argument('--output_csv',     required=True, help='Pfad für die Ausgabedatei der 2D-Positionen (CSV)')

    # Größen/FPS (weil wir kein Video öffnen): Original- & Zielgröße für das frühere Resizing
    parser.add_argument('--ori_w', type=int, default=960, help='Breite des Originalvideos (Pixel).')
    parser.add_argument('--ori_h', type=int, default=540, help='Höhe des Originalvideos (Pixel).')
    parser.add_argument('--out_w', type=int, default=960, help='Verwendete Zielbreite bei Homographie/Tracking (Pixel).')
    parser.add_argument('--out_h', type=int, default=540, help='Verwendete Zielhöhe bei Homographie/Tracking (Pixel).')
    parser.add_argument('--fps',   type=float, default=25.0, help='FPS für 1€-Filter.')

    # 1€-Filter Anti-„Reinfliegen“
    parser.add_argument('--gap_seconds', type=float, default=2.0,
                        help='Reset, wenn Tracklet >= gap_seconds nicht gesehen wurde.')
    parser.add_argument('--jump_reset_thresh', type=float, default=8.0,
                        help='Reset bei Positionssprung (Meter auf Pitch) >= jump_reset_thresh.')
    parser.add_argument('--prune_after_seconds', type=float, default=15.0,
                        help='Tracklets aus Speicher löschen, wenn seit so vielen Sekunden nicht gesehen.')

    args = parser.parse_args()

    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Tracking laden
    df_track = pd.read_csv(
        args.tracking,
        header=None,
        names=['frame','id','x','y','w','h','conf','class','jersey'],
        dtype={'frame': int, 'id': int, 'x': float, 'y': float, 'w': float, 'h': float,
               'conf': float, 'class': str, 'jersey': float}
    )
    df_track['jersey'] = df_track['jersey'].fillna(-1).astype(int)

    # Homographien laden
    df_h = pd.read_csv(args.transforms_csv, index_col='frame')
    H_cols = ['H00','H01','H02','H10','H11','H12','H20','H21','H22']
    for c in H_cols:
        if c not in df_h.columns:
            raise ValueError(f"Spalte {c} fehlt in {args.transforms_csv}")

    # Skalierung wie zuvor (wir hatten Frames auf out_w/out_h resized)
    sx = args.out_w / max(1, args.ori_w)
    sy = args.out_h / max(1, args.ori_h)

    # Pitch-Maße (wie vorher)
    template_w, template_h = 105.0, 68.0

    # Filter + Re-Entry-State
    filters   = {}      # (class,id) -> {'x': OneEuroFilter, 'y': OneEuroFilter}
    last_seen = {}      # (class,id) -> letzter frame_id (int)
    last_pos  = {}      # (class,id) -> (mx, my) in Pitchkoordinaten (zur Jump-Erkennung)

    def make_filters_for_key():
        return {
            'x': OneEuroFilter(freq=args.fps, mincutoff=0.8, beta=0.004),
            'y': OneEuroFilter(freq=args.fps, mincutoff=0.8, beta=0.004),
        }

    def need_gap_reset(key, frame_id):
        if key not in last_seen:
            return True
        gap_frames = frame_id - last_seen[key]
        return gap_frames >= int(round(args.gap_seconds * args.fps))

    def need_jump_reset(key, mx, my):
        if key not in last_pos:
            return False
        px, py = last_pos[key]
        dist = ((mx - px)**2 + (my - py)**2) ** 0.5
        return dist >= args.jump_reset_thresh

    # Timing
    frame_times = []
    total_frames = 0

    # Output CSV
    with open(args.output_csv, 'w') as csv_file:
        csv_file.write('frame,id,model_x,model_y,class,jersey\n')

        # wir iterieren über die Frames, die im Tracking vorkommen (sorted)
        for frame_id in sorted(df_track['frame'].unique()):
            t0 = perf_counter()

            # Zeilen dieses Frames (Achtung: viele Tracker 1-basiert; hier nehmen wir die Werte wie gegeben)
            rows = df_track[df_track['frame'] == frame_id]

            # Homographie holen (falls nicht vorhanden -> Identität)
            if frame_id in df_h.index:
                row = df_h.loc[frame_id, H_cols].values.astype(np.float32)
                H = row.reshape(3, 3)
            else:
                H = np.eye(3, dtype=np.float32)

            # Inverse H robust
            try:
                H_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                H_inv = np.eye(3, dtype=np.float32)

            # Pro Detection: Bild->Pitch transformieren + glätten
            for _, det in rows.iterrows():
                key = (str(det['class']), int(det['id']))
                if key not in filters:
                    filters[key] = make_filters_for_key()

                # Fußpunkt im (resized) Bild
                cx = (det['x'] + det['w'] / 2.0) * sx
                cy = (det['y'] + det['h'])       * sy
                pt = np.array([[[cx, cy]]], dtype=np.float32)

                # nach Modell (Pitch) in relativen Koordinaten
                mx_rel, my_rel = cv2.perspectiveTransform(pt, H_inv)[0, 0]

                # Weltursprung Mitte: in Pitch-Meter (0..105, 0..68)
                mx = mx_rel + template_w / 2.0
                my = -my_rel + template_h / 2.0

                # Anti-„Reinfliegen“
                gap_reset  = need_gap_reset(key, frame_id)
                jump_reset = need_jump_reset(key, mx, my)

                if gap_reset or jump_reset:
                    filters[key] = make_filters_for_key()
                    mx_smooth = mx
                    my_smooth = my
                    _ = filters[key]['x'](mx)  # prime
                    _ = filters[key]['y'](my)
                else:
                    mx_smooth = filters[key]['x'](mx)
                    my_smooth = filters[key]['y'](my)

                last_seen[key] = frame_id
                last_pos[key]  = (mx, my)

                cls = str(det['class'])
                jersey = int(det['jersey']) if pd.notna(det['jersey']) else -1

                csv_file.write(f"{frame_id},{int(det['id'])},{mx_smooth:.2f},{my_smooth:.2f},{cls},{jersey}\n")

            # Pruning alter Keys
            prune_frames = int(round(args.prune_after_seconds * args.fps))
            if prune_frames > 0:
                to_del = []
                for k, last_f in list(last_seen.items()):
                    if frame_id - last_f >= prune_frames:
                        to_del.append(k)
                for k in to_del:
                    last_seen.pop(k, None)
                    filters.pop(k, None)
                    last_pos.pop(k, None)

            t1 = perf_counter()
            dt = t1 - t0
            frame_times.append(dt)
            total_frames += 1

            # --- NEU: per-Frame Ausgabe ---
            print(f"[Timing] frame {frame_id}: {dt*1000:.2f} ms")

    # Timing-Report
    if frame_times:
        avg = sum(frame_times) / len(frame_times)
        fps_eff = (1.0 / avg) if avg > 0 else 0.0
        print(f"[Timing] Processed {total_frames} frames | avg {avg*1000:.2f} ms/frame ({fps_eff:.2f} FPS)")
    else:
        print("[Timing] No frames processed.")

if __name__ == '__main__':
    main()
