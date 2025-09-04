import os
import cv2
import yaml
import torch
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter  # NEW

from tqdm import tqdm

from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l

from utils.utils_calib import FramebyFrameCalib, pan_tilt_roll_to_orientation
from utils.utils_heatmap import (
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
    complete_keypoints,
    coords_to_dict,
)

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# ================== Konstanten ==================
TARGET_H, TARGET_W = 540, 960  # Model-Inputgröße

lines_coords = [
    [[0., 54.16, 0.], [16.5, 54.16, 0.]],
    [[16.5, 13.84, 0.], [16.5, 54.16, 0.]],
    [[16.5, 13.84, 0.], [0., 13.84, 0.]],
    [[88.5, 54.16, 0.], [105., 54.16, 0.]],
    [[88.5, 13.84, 0.], [88.5, 54.16, 0.]],
    [[88.5, 13.84, 0.], [105., 13.84, 0.]],
    [[0., 37.66, -2.44], [0., 30.34, -2.44]],
    [[0., 37.66, 0.], [0., 37.66, -2.44]],
    [[0., 30.34, 0.], [0., 30.34, -2.44]],
    [[105., 37.66, -2.44], [105., 30.34, -2.44]],
    [[105., 30.34, 0.], [105., 30.34, -2.44]],
    [[105., 37.66, 0.], [105., 37.66, -2.44]],
    [[52.5, 0., 0.], [52.5, 68, 0.]],
    [[0., 68., 0.], [105., 68., 0.]],
    [[0., 0., 0.], [0., 68., 0.]],
    [[105., 0., 0.], [105., 68., 0.]],
    [[0., 0., 0.], [105., 0., 0.]],
    [[0., 43.16, 0.], [5.5, 43.16, 0.]],
    [[5.5, 43.16, 0.], [5.5, 24.84, 0.]],
    [[5.5, 24.84, 0.], [0., 24.84, 0.]],
    [[99.5, 43.16, 0.], [105., 43.16, 0.]],
    [[99.5, 43.16, 0.], [99.5, 24.84, 0.]],
    [[99.5, 24.84, 0.], [105., 24.84, 0.]],
]

# =============== GPU-Preprocessing ===============
def frame_to_tensor_gpu(frame_bgr, device):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float().div_(255.0).unsqueeze(0)

    # erst “native” Zielgröße (z.B. 480×854)
    tgt_h, tgt_w = TARGET_H, TARGET_W
    t = torch.nn.functional.interpolate(t, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)

    # auf /32 padden (rechts/unten)
    def pad_to(x, m=32):
        _, _, h, w = x.shape
        pad_h = (m - (h % m)) % m
        pad_w = (m - (w % m)) % m
        if pad_h or pad_w:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))  # (left,right,top,bottom)
        return x

    t = pad_to(t, 32).to(device, non_blocking=True)
    return t


# =============== Projektion / Homographie ===============
def projection_from_cam_params(final_params_dict):
    cam_params = final_params_dict["cam_params"]
    x_focal_length = cam_params['x_focal_length']
    y_focal_length = cam_params['y_focal_length']
    principal_point = np.array(cam_params['principal_point'])
    position_meters = np.array(cam_params['position_meters'])
    rotation = np.array(cam_params['rotation_matrix'])

    It = np.eye(4)[:-1]
    It[:, -1] = -position_meters
    Q = np.array([
        [x_focal_length, 0, principal_point[0]],
        [0, y_focal_length, principal_point[1]],
        [0, 0, 1],
    ])
    P = Q @ (rotation @ It)
    return P


# =============== Batch-Inferenz ===============
def inference_batch(cam, frames_bgr, model, model_l, kp_threshold, line_threshold, pnl_refine, device):
    """
    frames_bgr: List[np.ndarray BGR], Länge = B
    Rückgabe: list[final_params_dict or None] in gleicher Reihenfolge
    """
    if len(frames_bgr) == 0:
        return []

    if device.startswith('cuda'):
        # Staple alle Frames zu einem Batch
        batch = torch.cat([frame_to_tensor_gpu(f, device) for f in frames_bgr], dim=0)  # (B,3,540,960)
        with torch.no_grad(), torch.cuda.amp.autocast():
            heatmaps = model(batch)     # (B, Ck, H, W)
            heatmaps_l = model_l(batch) # (B, Cl, H, W)
    else:
        # CPU-Pfad (langsamer, ohne AMP)
        import torchvision.transforms.functional as F
        from PIL import Image
        tensors = []
        for f_bgr in frames_bgr:
            rgb = cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            t = F.to_tensor(pil).float().unsqueeze(0)
            if t.size(-1) != TARGET_W:
                t = torch.nn.functional.interpolate(
                    t, size=(TARGET_H, TARGET_W), mode="bilinear", align_corners=False
                )
            tensors.append(t)
        batch = torch.cat(tensors, dim=0).to(device)
        with torch.no_grad():
            heatmaps = model(batch)
            heatmaps_l = model_l(batch)

    # Postprocessing pro Sample
    B = batch.size(0)
    results = []
    Ht, Wt = TARGET_H, TARGET_W

    # Keypoints/Linien aus Heatmaps holen (Funktionen erwarten Batch)
    kp_coords_all   = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
    line_coords_all = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1, :, :])

    for i in range(B):
        kp_dict    = coords_to_dict(kp_coords_all[i:i+1], threshold=kp_threshold)[0]
        lines_dict = coords_to_dict(line_coords_all[i:i+1], threshold=line_threshold)[0]
        kp_dict, lines_dict = complete_keypoints(kp_dict, lines_dict, w=Wt, h=Ht, normalize=True)

        cam.update(kp_dict, lines_dict)
        final_params_dict = cam.heuristic_voting(refine_lines=pnl_refine)
        results.append(final_params_dict)

    return results  # list length B


# =============== Hauptprozess ===============
def process_input(input_path, input_type, model, model_l, kp_threshold, line_threshold, pnl_refine,
                  save_path, display, device, batch_size):

    cap = cv2.VideoCapture(input_path)
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25

    cam = FramebyFrameCalib(iwidth=TARGET_W, iheight=TARGET_H, denormalize=True)

    # --- timing stats ---  # NEW
    frame_times = []      # seconds per frame
    total_processed = 0   # how many frames we actually processed

    def handle_batch(buf_frames, buf_indices):  # NEW
        nonlocal frame_times, total_processed
        if len(buf_frames) == 0:
            return

        t0 = perf_counter()
        results = inference_batch(cam, buf_frames, model, model_l,
                                  kp_threshold, line_threshold, pnl_refine, device)
        t1 = perf_counter()

        per_frame = (t1 - t0) / len(buf_frames)
        frame_times.extend([per_frame] * len(buf_frames))
        total_processed += len(buf_frames)

        for (idx, final_params_dict) in zip(buf_indices, results):
            if final_params_dict is not None:
                P = projection_from_cam_params(final_params_dict)
                H = P[:, [0, 1, 3]]
                H = H / H[2, 2]
            else:
                H = np.eye(3); H = H / H[2, 2]
            if writer is not None:
                writer.writerow([idx] + H.flatten().tolist())

    if input_type == 'video':
        if save_path != "":
            out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
            csv_path = os.path.splitext(save_path)[0] + f"{os.path.basename(input_path).split('.')[0]}_homographies.csv"
            csv_file = open(csv_path, 'w', newline='')
            writer = csv.writer(csv_file)
            writer.writerow(['frame'] + [f'H{i}{j}' for i in range(3) for j in range(3)])
        else:
            out = None
            writer = None

        pbar = tqdm(total=total_frames)
        frame_idx = 0

        # Batch-Buffer
        buf_frames = []   # rohe BGR-Frames, für spätere Ausgabe/Projektion
        buf_indices = []  # Frame-Indices, damit Reihenfolge klar bleibt

        while True:
            ret, frame = cap.read()
            if not ret:
                # Rest-Buffer flushen
                handle_batch(buf_frames, buf_indices)  # NEW
                break

            buf_frames.append(frame)
            buf_indices.append(frame_idx)

            # Wenn Buffer voll -> Batch-Inferenz
            if (device.startswith('cuda') and len(buf_frames) == batch_size) or \
               (not device.startswith('cuda') and len(buf_frames) == 1):
                handle_batch(buf_frames, buf_indices)  # NEW
                buf_frames.clear()
                buf_indices.clear()

            pbar.update(1)
            frame_idx += 1

        cap.release()
        if out is not None:
            out.release()
        if writer is not None:
            csv_file.close()

        # --- timing summary ---  # NEW
        if frame_times:
            avg = sum(frame_times) / len(frame_times)
            fps_eff = (1.0 / avg) if avg > 0 else 0.0
            print(f"[Timing] Processed {total_processed} frames | avg {avg:.4f} s/frame ({fps_eff:.2f} FPS)")
        else:
            print("[Timing] No frames processed.")

        cv2.destroyAllWindows()

    elif input_type == 'image':
        frame = cv2.imread(input_path)
        if frame is None:
            print(f"Error: Unable to read the image {input_path}")
            return

        t0 = perf_counter()  # NEW
        results = inference_batch(cam, [frame], model, model_l,
                                  kp_threshold, line_threshold, pnl_refine, device)
        t1 = perf_counter()  # NEW
        print(f"[Timing] Single image processing time: {(t1 - t0):.4f} s")  # NEW

        final_params_dict = results[0]
        if final_params_dict is not None:
            P = projection_from_cam_params(final_params_dict)
            projected_frame = frame  # oder project(frame, P) falls du die Linien zeichnen willst
        else:
            projected_frame = frame

        if save_path != "":
            cv2.imwrite(save_path, projected_frame)
        else:
            plt.imshow(cv2.cvtColor(projected_frame, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()


# ================== Main ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video or image and plot lines on each frame.")
    parser.add_argument("--weights_kp", type=str, required=True, help="Path to the model for keypoint inference.")
    parser.add_argument("--weights_line", type=str, required=True, help="Path to the model for line projection.")
    parser.add_argument("--kp_threshold", type=float, default=0.3434, help="Threshold for keypoint detection.")
    parser.add_argument("--line_threshold", type=float, default=0.7867, help="Threshold for line detection.")
    parser.add_argument("--pnl_refine", action="store_true", help="Enable PnL refinement module.")
    parser.add_argument("--device", type=str, default="cuda:0", help="CPU or CUDA device index")
    parser.add_argument("--batch_size", type=int, default=8, help="Batchgröße für CUDA (bei CPU ignoriert).")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input video or image file.")
    parser.add_argument("--input_type", type=str, choices=['video', 'image'], required=True,
                        help="Type of input: 'video' or 'image'.")
    parser.add_argument("--save_path", type=str, default="", help="Path to save the processed video (optional).")
    parser.add_argument("--display", action="store_true", help="Enable real-time display.")
    args = parser.parse_args()

    input_path     = args.input_path
    input_type     = args.input_type
    weights_kp     = args.weights_kp
    weights_line   = args.weights_line
    pnl_refine     = args.pnl_refine
    save_path      = args.save_path
    device         = args.device
    batch_size     = max(1, args.batch_size if device.startswith('cuda') else 1)
    display        = args.display and input_type == 'video'
    kp_threshold   = args.kp_threshold
    line_threshold = args.line_threshold

    cfg   = yaml.safe_load(open("config/hrnetv2_w48.yaml", 'r'))
    cfg_l = yaml.safe_load(open("config/hrnetv2_w48_l.yaml", 'r'))

    loaded_state   = torch.load(weights_kp,   map_location=device)
    loaded_state_l = torch.load(weights_line, map_location=device)

    model   = get_cls_net(cfg)
    model_l = get_cls_net_l(cfg_l)
    model.load_state_dict(loaded_state)
    model_l.load_state_dict(loaded_state_l)

    model.to(device)
    model_l.to(device)
    # WICHTIG: kein model.half(); wir nutzen AMP (autocast) für Stabilität/Kompatibilität
    model.eval()
    model_l.eval()

    process_input(input_path, input_type, model, model_l, kp_threshold, line_threshold, pnl_refine,
                  save_path, display, device, batch_size)
