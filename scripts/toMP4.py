#!/usr/bin/env python3
import cv2
import os
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Convert a directory of images to an MP4 video"
    )
    parser.add_argument(
        "-i", "--input_dir", required=True,
        help="Directory containing image frames (e.g., .jpg files)"
    )
    parser.add_argument(
        "-o", "--output", default="output.mp4",
        help="Output video file path (MP4) or directory to save video"
    )
    parser.add_argument(
        "-f", "--fps", type=float, default=25.0,
        help="Frame rate (FPS) for the output video"
    )
    parser.add_argument(
        "-e", "--extension", default=".jpg",
        help="Image file extension to include (default: .jpg)"
    )
    args = parser.parse_args()

    # Validate input directory
    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    # Determine output file path
    out = args.output
    # If output is existing directory, use input folder name as video name
    if os.path.isdir(out):
        base = os.path.basename(os.path.normpath(input_dir))
        out_file = os.path.join(out, f"{base}.mp4")
    else:
        # If output ends with os.sep or has no extension, treat as directory
        if out.endswith(os.sep):
            os.makedirs(out, exist_ok=True)
            base = os.path.basename(os.path.normpath(input_dir))
            out_file = os.path.join(out, f"{base}.mp4")
        else:
            # ensure parent directory exists
            parent = os.path.dirname(out)
            if parent and not os.path.isdir(parent):
                print(f"Error: Directory '{parent}' does not exist.")
                sys.exit(1)
            out_file = out if out.lower().endswith('.mp4') else out + '.mp4'

    # Collect and sort image files
    files = [f for f in os.listdir(input_dir)
             if f.lower().endswith(args.extension.lower())]
    if not files:
        print(f"No files with extension '{args.extension}' found in {input_dir}")
        sys.exit(1)
    files.sort()

    # Read first image to get dimensions
    first_path = os.path.join(input_dir, files[0])
    frame = cv2.imread(first_path)
    if frame is None:
        print(f"Failed to read the first image: {first_path}")
        sys.exit(1)
    height, width = frame.shape[:2]

    # Setup video writer with MP4 codec
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(out_file, fourcc, args.fps, (width, height))

    # Write frames to video
    for fname in files:
        path = os.path.join(input_dir, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not read {path}, skipping")
            continue
        # Resize if dimensions differ
        if (img.shape[1], img.shape[0]) != (width, height):
            img = cv2.resize(img, (width, height))
        writer.write(img)

    writer.release()
    print(f"Video saved to {out_file}")

if __name__ == "__main__":
    main()
