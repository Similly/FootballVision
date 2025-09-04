#!/usr/bin/env python3
import cv2
import os
import argparse
import sys
import re
from pathlib import Path
from typing import List, Tuple

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def find_frame_files(d: Path, ext: str) -> List[Path]:
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ext.lower()],
                  key=lambda p: natural_key(p.name))

def try_open_writer(out_file: str, fps: float, size: Tuple[int, int], prefer=None):
    if prefer is None:
        prefer = ['avc1', 'H264', 'X264', 'mp4v', 'MJPG']
    for code in prefer:
        fourcc = cv2.VideoWriter_fourcc(*code)
        writer = cv2.VideoWriter(out_file, fourcc, fps, size)
        if writer.isOpened():
            print(f"[info] Using codec '{code}' -> {out_file}")
            return writer, code
        writer.release()
    return None, None

def write_video_from_frames(frames: List[Path], out_file: str, fps: float) -> bool:
    if not frames:
        print(f"[warn] No frames to write for {out_file}")
        return False

    first = cv2.imread(str(frames[0]))
    if first is None:
        print(f"[error] Failed to read first frame: {frames[0]}")
        return False
    h, w = first.shape[:2]

    writer, used = try_open_writer(out_file, fps, (w, h))
    if writer is None:
        print("[error] Could not open VideoWriter with any codec (tried avc1/H264/X264/mp4v/MJPG).")
        return False

    written = 0
    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            print(f"[warn] Could not read {f}, skipping.")
            continue
        if (img.shape[1], img.shape[0]) != (w, h):
            img = cv2.resize(img, (w, h))
        writer.write(img)
        written += 1

    writer.release()
    if written == 0:
        print(f"[warn] Wrote 0 frames for {out_file}")
        return False

    print(f"[ok] Saved {out_file} ({written} frames @ {fps} fps)")
    return True

def extract_seq_num(seq_dir: Path, root: Path) -> str:
    """
    Nimmt den *Top-Level*-Ordner unterhalb von root (z.B. 'challenge/XYZ...')
    und extrahiert dessen letzte drei Ziffern.
    """
    try:
        rel = seq_dir.relative_to(root)
        top = rel.parts[0] if rel.parts else seq_dir.name
    except ValueError:
        # Falls seq_dir nicht unter root liegt (sollte nicht passieren), fallback auf eigenen Namen
        top = seq_dir.name

    m = re.search(r'(\d{3})$', top)
    if m:
        return m.group(1)

    # Fallback: irgendeine 3er-Zifferngruppe aus dem Top-Level-Namen nehmen (letzte)
    matches = re.findall(r'(\d{3})', top)
    if matches:
        return matches[-1]

    # Letzter Fallback: die letzten 3 Zeichen (nicht ideal, aber verhindert Crash)
    print(f"[warn] Could not find 3 trailing digits in '{top}'. Using last 3 chars as name.")
    return top[-3:]

def main():
    parser = argparse.ArgumentParser(
        description="Convert image sequences to MP4. If INPUT is a directory with subfolders, each subfolder with frames becomes one video. Output filenames use only the 3-digit sequence number."
    )
    parser.add_argument(
        "-i", "--input_dir", required=True,
        help="Directory containing frames or subdirectories with frames (e.g., challenge/)"
    )
    parser.add_argument(
        "-o", "--output", default="output_videos",
        help="Output .mp4 file OR directory to place multiple videos (default: output_videos)"
    )
    parser.add_argument(
        "-f", "--fps", type=float, default=25.0,
        help="Frame rate (FPS) for output video(s)"
    )
    parser.add_argument(
        "-e", "--extension", default=".jpg",
        help="Image file extension to include (default: .jpg)"
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true",
        help="Scan subfolders recursively for sequences"
    )
    args = parser.parse_args()

    root = Path(args.input_dir)
    if not root.is_dir():
        print(f"[error] Input directory '{root}' does not exist.")
        sys.exit(1)

    out_arg = Path(args.output)

    # Prüfen, ob root selbst Frames enthält (Single-Sequence) oder wir Subdirs sammeln (Multi-Sequence)
    root_frames = find_frame_files(root, args.extension)

    sequences = []
    if root_frames:
        # Single sequence (root hat Frames)
        sequences = [(root, root_frames)]
        out_is_file = out_arg.suffix.lower() == ".mp4"
        if not out_is_file:
            out_arg.mkdir(parents=True, exist_ok=True)
    else:
        # Multi sequence: Subdirs sammeln (rekursiv optional)
        cand_dirs = [d for d in (root.rglob("*") if args.recursive else root.iterdir()) if d.is_dir()]
        for d in cand_dirs:
            frames = find_frame_files(d, args.extension)
            if frames:
                sequences.append((d, frames))
        if not sequences:
            print(f"[error] No frames with extension '{args.extension}' found under '{root}'.")
            sys.exit(1)

        # Bei mehreren Sequenzen muss output ein Verzeichnis sein
        if out_arg.suffix.lower() == ".mp4":
            print("[error] Output points to a single file but multiple sequences were found. "
                  "Provide an output directory instead.")
            sys.exit(1)
        out_arg.mkdir(parents=True, exist_ok=True)

    print(f"[info] Found {len(sequences)} sequence(s).")

    for seq_dir, frames in sequences:
        # Output wählen:
        if out_arg.suffix.lower() == ".mp4":
            # Single-Sequence: falls User explizit eine Datei angibt, respektieren wir das
            out_file = str(out_arg)
        else:
            # Multi-Sequence (oder Single in Dir): NUR 3-stellige Sequenznummer nutzen
            seq_num = extract_seq_num(seq_dir, root)
            out_file = str(out_arg / f"{seq_num}.mp4")

        ok = write_video_from_frames(frames, out_file, args.fps)
        if not ok:
            print(f"[warn] Failed to write video for sequence: {seq_dir}")

if __name__ == "__main__":
    main()
