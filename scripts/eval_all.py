# run_all_videos.py
import os, glob, subprocess
from pathlib import Path
import argparse

VIDEO_EXTS = (".mkv", ".mp4", ".avi", ".mov")

def find_videos(root):
    vids = []
    for ext in VIDEO_EXTS:
        vids += glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True)
    return sorted(vids)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_root", required=True, help="Folder with videos (recurses).")
    ap.add_argument("--results_dir", default="output/soccernet_mot", help="Where to write MOT .txt files.")
    ap.add_argument("--script", default="testScripts/det_and_tracking_eval.py")
    ap.add_argument("--python", default="python", help="Python executable to use.")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    videos = find_videos(args.videos_root)
    if not videos:
        print("No videos found.")
        return

    for i, vid in enumerate(videos, 1):
        seq = Path(vid).stem
        out_txt = os.path.join(args.results_dir, f"{seq}.txt")

        env = os.environ.copy()
        env["SN_RESULTS_TXT"] = out_txt  # your script reads this (fallbacks if unset)

        print(f"[{i}/{len(videos)}] {seq}")
        # Call your script once per video → fresh process, fresh globals
        subprocess.run([args.python, args.script, vid], check=True, env=env)

    print(f"Done. MOT results in: {args.results_dir}")
    print("Zip them for EvalAI, e.g.:  zip -j submission.zip", os.path.join(args.results_dir, "*.txt"))

if __name__ == "__main__":
    main()
