import os
import sys
import time as ti
from collections import defaultdict
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from sklearn.cluster import KMeans

from .config import (
    DRAW_ROI, ROI_COLORS, ROI_THICK,
    LEGIBILITY_ENABLED, LEG_THR,
    PARSEQ_CKPT, CONF_MIN,
    AREA_GROWTH_TRIG, OCR_FRAME_GAP,
    COLORS, TARGET_CLASSES
)
from .teamcolor import get_grass_color, extract_kit_color, KitsHolder
from .ocr_parseq import load_parseq_from_ckpt, jersey_rois_with_abs, read_number_from_roi
from .legibility import LegibilityClassifier34, is_legible
from .trackers_init import build_trackers
from .voting import (
    track_history, get_dominant_team, add_vote, maybe_assign_number, leaky_decay_all,
    track_assigned_num, last_ocr_frame, last_box_area, vote_sum_offline
)
from .postprocess import postprocess_and_write

def main():
    # Args
    if len(sys.argv) < 2:
        print("Usage: python -m tracking_pipeline.main <input_video.mp4>")
        sys.exit(1)
    video_path = sys.argv[1]

    output_dir = "outputVids/track_mot"
    os.makedirs(output_dir, exist_ok=True)

    # Video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Fehler: Video {video_path} konnte nicht geöffnet werden.")
        sys.exit(1)

    # Video params
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Device & YOLO
    print(f"MPS Available: {torch.backends.mps.is_available()}, Built: {torch.backends.mps.is_built()}")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("using device:", device)

    model = YOLO("detModels/yolov8n+SoccerNet5class_phase2/weights/best.pt")
    model.to(device.type)
    print(model.names)

    # first frame for grass color
    ret, first_frame = cap.read()
    if not ret:
        print("Fehler: Konnte ersten Frame nicht lesen.")
        sys.exit(1)
    _hsv, grass_mask = get_grass_color(first_frame)
    grass_hsv_val = cv2.mean(cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV), mask=grass_mask)[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Writers / outputs
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    output_video = os.path.join(output_dir, f"output_video_OCR.mp4")
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    mot_file_path = os.path.join(output_dir, "mot_results.txt")
    output_rows = []

    # Trackers init (keeps your env hack & behavior)
    if os.environ.get('CUDA_VISIBLE_DEVICES', '').strip().lower() == 'cuda':
        os.environ.pop('CUDA_VISIBLE_DEVICES')

    has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    if has_cuda:
        torch.cuda.set_device(0)
        device = torch.device('cuda:0')
        mot_device = '0'
    else:
        device = torch.device('cpu')
        mot_device = 'cpu'

    print("PyTorch sees CUDA:", torch.cuda.is_available(), "num:", torch.cuda.device_count(), "device:", device)
    trackers_dict = build_trackers(device, mot_device, fps)

    # OCR (PARSeq)
    parseq_sys, parseq_tf = load_parseq_from_ckpt(PARSEQ_CKPT, device)
    ocr = parseq_sys

    # Legibility
    leg_device = device
    leg_model = LegibilityClassifier34()
    leg_ckpt = "legibility_resnet34_soccer_20240215.pth"
    if os.path.exists(leg_ckpt):
        sd = torch.load(leg_ckpt, map_location=leg_device)
        leg_model.load_state_dict(sd, strict=True)
    else:
        print("[Legibility] Warnung: kein Checkpoint gefunden – nur ImageNet-Pretrain.")
    leg_model.to(leg_device).eval()

    # Main loop
    start_time = ti.time()
    detection_time = 0.0
    track_time = 0.0
    ocr_time = 0.0
    ocr_calls = 0
    ocr_roi_calls = 0
    frame_idx = 0

    kits_clf = None
    left_label = None
    old_left_center = None
    old_right_center = None

    while True:
        frame_tic = ti.time()
        frame_ocr_time = 0.0

        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        print(f"\n{frame_idx}")
        cv2.putText(frame, f"Frame {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Leaky decay
        leaky_decay_all()

        # Detection
        t0 = ti.time()
        results = model(frame)
        frame_detection_time = ti.time() - t0
        detection_time += frame_detection_time

        # Collect dets
        dets = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0]); cls = int(box.cls[0])
            dets.append([x1, y1, x2, y2, conf, cls])
        dets = np.array(dets)

        # Per-class tracking
        t1 = ti.time()
        results_by_class = {}
        for class_id in TARGET_CLASSES:
            class_dets = dets[dets[:, 5] == class_id] if dets.size else np.empty((0, 6))
            online_targets = trackers_dict[class_id].update(class_dets, frame)
            results_by_class[class_id] = online_targets
        frame_track_time = ti.time() - t1
        track_time += frame_track_time

        # Non-player drawing (Ball, Ref)
        for cid in [2, 3]:
            for t in results_by_class[cid]:
                x1, y1, x2, y2, _track_id = map(int, t[:5])
                w, h = x2 - x1, y2 - y1
                col = COLORS[cid]
                shown_id = 1  # single id for these

                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cv2.putText(frame, f"C{cid} T{shown_id}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

                output_rows.append((frame_idx, shown_id, float(x1), float(y1),
                                    float(w), float(h), -1.0, str(cid), -1))

        # Goalkeepers (ID=1 + side)
        for t in results_by_class[1]:
            x1, y1, x2, y2, _track_id = map(int, t[:5])
            w, h = x2 - x1, y2 - y1
            side = 'L' if x1 < width / 2 else 'R'
            col = (0, 255, 0) if side == 'L' else (0, 128, 0)
            shown_id = 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame, f"GK-{side} ID{shown_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

            extra_str = f"GK-{side}"
            output_rows.append((frame_idx, shown_id, float(x1), float(y1),
                                float(w), float(h), -1.0, extra_str, -1))

        # Players (class 0): kit colors & clustering
        players = results_by_class[0]
        kit_colors = []
        bboxes = []
        for t in players:
            x1, y1, x2, y2, track_id = map(int, t[:5])
            crop = frame[y1:y2, x1:x2]
            color_vec = extract_kit_color(crop, grass_hsv_val)
            kit_colors.append(color_vec)
            bboxes.append((x1, y1, x2, y2, track_id))

        # periodic re-fit
        if frame_idx == 1 or frame_idx % 125 == 0:
            if len(kit_colors) >= 2:
                new_clf = KMeans(n_clusters=2, random_state=42).fit(kit_colors)
                new_centers = new_clf.cluster_centers_
                if old_right_center is not None:
                    dists = [abs(c[0] - old_right_center[0]) for c in new_centers]
                    new_right_idx = int(np.argmin(dists))
                    new_left_idx = 1 - new_right_idx
                    reordered = np.array([new_centers[new_left_idx], new_centers[new_right_idx]])
                    kits_clf = KitsHolder(reordered)
                    old_left_center, old_right_center = reordered
                else:
                    centers_x = [(x1 + x2) / 2 for (x1, y1, x2, y2, _) in bboxes]
                    labels_ref = new_clf.labels_
                    avg_x = [np.mean([cx for cx, lbl in zip(centers_x, labels_ref) if lbl == i]) for i in [0, 1]]
                    right_idx = int(np.argmax(avg_x))
                    left_idx = 1 - right_idx
                    reordered = np.array([new_clf.cluster_centers_[left_idx], new_clf.cluster_centers_[right_idx]])
                    kits_clf = KitsHolder(reordered)
                    old_left_center, old_right_center = reordered
                    left_label = 0

        # Draw players & OCR
        if kits_clf is not None and kit_colors:
            labels = kits_clf.predict(kit_colors)
            for lbl, (x1, y1, x2, y2, track_id) in zip(labels, bboxes):
                team_label = 'L' if lbl == left_label else 'R'
                track_history[track_id].append(team_label)
                dom_team = get_dominant_team(track_history[track_id])
                col = (255, 0, 0) if dom_team == 'L' else (0, 0, 255)

                box_w, box_h = x2 - x1, y2 - y1
                area = float(box_w * box_h)
                do_ocr = False

                if frame_idx - last_ocr_frame[track_id] >= OCR_FRAME_GAP:
                    do_ocr = True
                if area > last_box_area[track_id] * AREA_GROWTH_TRIG:
                    do_ocr = True
                last_box_area[track_id] = max(last_box_area[track_id], area)

                if do_ocr:
                    ocr_calls += 1
                    t2 = ti.time()
                    player_crop = frame[y1:y2, x1:x2]
                    leg_ok, leg_p = (True, 1.0)
                    if LEGIBILITY_ENABLED:
                        leg_ok, leg_p = is_legible(player_crop, leg_model, leg_device, thr=LEG_THR)

                    if DRAW_ROI and LEGIBILITY_ENABLED:
                        leg_col = (0, 255, 0) if leg_ok else (0, 0, 255)
                        leg_y = min(frame.shape[0] - 4, y2 + 16)
                        cv2.putText(frame, f"leg {leg_p:.2f}",
                                    (x1, leg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, leg_col, 1)

                    if leg_ok:
                        for roi_img, (rx1, ry1, rx2, ry2), tag in jersey_rois_with_abs(frame, x1, y1, x2, y2):
                            if DRAW_ROI:
                                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), ROI_COLORS.get(tag, (200, 200, 200)), ROI_THICK)
                            ocr_roi_calls += 1
                            out = read_number_from_roi(roi_img, ocr, parseq_tf, device, conf_min=CONF_MIN)
                            if out is not None:
                                num, conf = out
                                add_vote(track_id, num, conf)
                                if DRAW_ROI:
                                    cv2.putText(frame, f"raw:{num} {conf:.2f}",
                                                (x1, max(0, y1 - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                print(f"[F{frame_idx} T{track_id} ROI=({rx1},{ry1},{rx2},{ry2})] #{num} conf={conf:.2f}")

                    if leg_ok:
                        last_ocr_frame[track_id] = frame_idx
                        maybe_assign_number(track_id)
                    frame_ocr_time += ti.time() - t2

                assigned = track_assigned_num.get(track_id, None)

                # Draw box/label
                label_txt = f"P-{dom_team} ID{track_id}"
                if assigned is not None:
                    label_txt += f" #{assigned}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cv2.putText(frame, label_txt, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                jersey_out = assigned if assigned is not None else -1
                output_rows.append((frame_idx, track_id, float(x1), float(y1), float(box_w), float(box_h), -1.0, dom_team, jersey_out))

        ocr_time += frame_ocr_time
        frame_time = ti.time() - frame_tic
        print(f"Frame {frame_idx} Zeit: {frame_time*1000:.1f}ms "
              f"(Det: {frame_detection_time*1000:.1f}ms, "
              f"Track: {frame_track_time*1000:.1f}ms, "
              f"OCR: {frame_ocr_time*1000:.1f}ms)")

        video_writer.write(frame)

    # Cleanup
    video_writer.release()
    cap.release()

    # Postprocess & write MOT-like file
    relabeled_rows = postprocess_and_write(output_rows, vote_sum_offline, mot_file_path)

    # Runtime summary (same math, now using frame_idx)
    total_time = ti.time() - start_time
    other_time = total_time - (detection_time + track_time + ocr_time)
    avg_det_ms  = (detection_time / max(1, frame_idx)) * 1000
    avg_trk_ms  = (track_time     / max(1, frame_idx)) * 1000
    avg_ocr_ms  = (ocr_time       / max(1, frame_idx)) * 1000
    avg_tot_ms  = (total_time     / max(1, frame_idx)) * 1000
    fps_eff     = (frame_idx / total_time) if total_time > 0 else 0.0

    print("\n=== Runtime Summary ===")
    print(f"Frames: {frame_idx} | Duration: {total_time:.2f}s | FPS: {fps_eff:.2f}")
    print(f"Avg per frame [ms]  ->  Det: {avg_det_ms:.2f} | Track: {avg_trk_ms:.2f} | OCR: {avg_ocr_ms:.2f} | Total: {avg_tot_ms:.2f}")
    print(f"Time share [%]       ->  Det: {100*detection_time/total_time:.1f} | Track: {100*track_time/total_time:.1f} | OCR: {100*ocr_time/total_time:.1f} | Other: {100*other_time/total_time:.1f}")
    print(f"OCR calls: players={ocr_calls}, roi_calls={ocr_roi_calls}")
    print(f"Video: {output_video}")
    print(f"MOT-ähnlicher Output mit Jersey-Spalte: {mot_file_path}")

if __name__ == "__main__":
    main()
