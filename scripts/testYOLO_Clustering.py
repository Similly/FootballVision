#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import time
from ultralytics import YOLO
from sklearn.cluster import KMeans
import os

# Label and color mappings
label_names = [
    "Player-L", "Player-R", "GK-L", "GK-R",
    "Ball", "Main Ref", "Side Ref", "Staff"
]
box_colors = {
    0: (150, 50, 50),   # Player-L
    1: (37, 47, 150),   # Player-R
    2: (41, 248, 165),  # GK-L
    3: (166, 196, 10),  # GK-R
    4: (155, 62, 157),  # Ball
    5: (123, 174, 213), # Main Ref
    6: (217, 89, 204),  # Side Ref
    7: (22, 11, 15)     # Staff
}

def get_grass_color(img):
    """
    Finds the color of the grass in the background of the image
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    grass_color = cv2.mean(img, mask=mask)
    return grass_color[:3]

def get_players_boxes(result):
    """
    Extracts player crops and their bounding boxes from YOLO result
    """
    players_imgs = []
    players_boxes = []
    for box in result.boxes:
        cls_id = int(box.cls)
        if cls_id == 0:  # class 0: player
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            player_img = result.orig_img[y1:y2, x1:x2]
            players_imgs.append(player_img)
            players_boxes.append(box)
    return players_imgs, players_boxes

def get_kits_colors(players, grass_hsv=None, frame=None):
    """
    Computes dominant kit color of each player crop
    """
    kits_colors = []
    if grass_hsv is None and frame is not None:
        grass_bgr = get_grass_color(frame)
        grass_hsv = cv2.cvtColor(
            np.uint8([[grass_bgr]]),
            cv2.COLOR_BGR2HSV
        )[0][0]
    for img in players:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        low = np.array([int(grass_hsv[0]) - 10, 40, 40])
        high = np.array([int(grass_hsv[0]) + 10, 255, 255])
        green_mask = cv2.inRange(hsv, low, high)
        inv_mask = cv2.bitwise_not(green_mask)
        half = np.zeros(inv_mask.shape, np.uint8)
        half[0:inv_mask.shape[0]//2, :] = 255
        mask = cv2.bitwise_and(inv_mask, half)
        color = cv2.mean(img, mask=mask)[:3]
        kits_colors.append(np.array(color))
    return kits_colors

def get_kits_classifier(kits_colors):
    """
    Fits a KMeans model to cluster kit colors into two teams
    """
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(kits_colors)
    return kmeans

def classify_kits(kits_classifier, kits_colors):
    """
    Predicts team index for each kit color
    """
    return kits_classifier.predict(kits_colors)

def get_left_team_label(players_boxes, kits_colors, kits_clf):
    """
    Determines which cluster corresponds to left team based on x-coordinate means
    """
    left_label = 0
    team0_x = []
    team1_x = []
    for i, box in enumerate(players_boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        team = classify_kits(kits_clf, [kits_colors[i]])[0]
        if team == 0:
            team0_x.append(x1)
        else:
            team1_x.append(x1)
    if team0_x and team1_x:
        if np.mean(team0_x) > np.mean(team1_x):
            left_label = 1
    return left_label

def annotate_video(input_path, model, output_path, out_fps=None):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = out_fps if out_fps else input_fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    start = time.time()
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # YOLO inference
        results = model(frame)[0]
        # Team clustering
        players_imgs, players_boxes = get_players_boxes(results)
        grass_hsv = get_grass_color(frame)
        kits_colors = get_kits_colors(players_imgs, grass_hsv, frame)
        if kits_colors:
            kits_clf = get_kits_classifier(kits_colors)
            left_team = get_left_team_label(players_boxes, kits_colors, kits_clf)
            team_preds = classify_kits(kits_clf, kits_colors)
        else:
            left_team = None
            team_preds = []
        # Draw annotations
        player_i = 0
        for box in results.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls_id in (0, 1):  # player or goalkeeper
                team = team_preds[player_i] if player_i < len(team_preds) else 0
                if cls_id == 0:
                    idx = 0 if team == left_team else 1
                else:
                    idx = 2 if team == left_team else 3
                player_i += 1
            else:
                idx = cls_id + 2  # map class 2->4, 3->5, 4->6, 5->7
            color = box_colors[idx]
            label = f"{label_names[idx]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)
        writer.write(frame)
        frame_idx += 1
    end = time.time()
    print(f"Processed {frame_idx} frames in {end-start:.2f}s.")
    print(f"Output saved to {output_path}")
    cap.release()
    writer.release()

def annotate_frames_folder(input_folder, model, output_path, fps):
    # Collect frame file names
    files = sorted([f for f in os.listdir(input_folder)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not files:
        raise ValueError(f"No frames in {input_folder}")

    # Initial frame for clustering
    first_frame = cv2.imread(os.path.join(input_folder, files[0]))
    res0 = model(first_frame)[0]
    players0, boxes0 = get_players_boxes(res0)
    grass_bgr = get_grass_color(first_frame)
    grass_hsv = cv2.cvtColor(np.uint8([[grass_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    if players0:
        kits_colors0 = get_kits_colors(players0, grass_hsv)
        kits_clf = get_kits_classifier(kits_colors0)
        left_team = get_left_team_label(boxes0, kits_colors0, kits_clf)
    else:
        kits_clf, left_team = None, None

    # Video writer based on first frame size
    h, w = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    start = time.time()
    for idx, fname in enumerate(files, 1):
        frame = cv2.imread(os.path.join(input_folder, fname))
        res = model(frame)[0]

        # classify using pre-fit classifier
        players, boxes = get_players_boxes(res)
        if kits_clf and players:
            colors = get_kits_colors(players, grass_hsv)
            teams = classify_kits(kits_clf, colors)
        else:
            teams = []

        # draw annotations
        pi = 0
        for box in res.boxes:
            cid = int(box.cls); conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cid in (0,1) and pi < len(teams):
                team = teams[pi]
                if cid == 0:
                    idx_c = 0 if team == left_team else 1
                else:
                    idx_c = 2 if team == left_team else 3
                pi += 1
            else:
                idx_c = cid + 2
            color = box_colors[idx_c]
            label = f"{label_names[idx_c]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        writer.write(frame)
        print(f"Frame {idx}/{len(files)} processed")

    dur = time.time() - start
    print(f"Processed {len(files)} frames in {dur:.2f}s")
    print(f"Output saved to {output_path}")
    writer.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO-based player detection and team clustering")
    parser.add_argument("-i", "--input", required=True, help="Path to input video (mp4)")
    parser.add_argument("-o", "--output", default="output.mp4", help="Path to save annotated video")
    parser.add_argument("-m", "--model", default="./weights/best.pt", help="Path to YOLO weights file")
    parser.add_argument("-f", "--fps", type=float, help="FPS for output video (default: same as input)")
    args = parser.parse_args()

    yolo_model = YOLO(args.model)
    #annotate_video(args.input, yolo_model, args.output, args.fps)
    annotate_frames_folder(args.input, yolo_model, args.output, args.fps)

