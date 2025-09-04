import os
import glob
import shutil

BASE_DIR    = '../data/tracking/train'
OUTPUT_DIR  = '../data/tracking/train_25pct'
STEP        = 4  # Behalte jedes STEP-te Frame

def downsample_clip(clip_dir, out_base, step):
    img_dir    = os.path.join(clip_dir, 'img1')
    lbl_dir    = os.path.join(clip_dir, 'labels')
    lbl5_dir    = os.path.join(clip_dir, 'labels5')
    clip_name  = os.path.basename(clip_dir)
    out_img    = os.path.join(out_base, clip_name, 'img1')
    out_lbl    = os.path.join(out_base, clip_name, 'labels')
    out_lbl5    = os.path.join(out_base, clip_name, 'labels5')

    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)
    os.makedirs(out_lbl5, exist_ok=True)


    # Ermittle Offset (erstes Frame-Index modulo STEP)
    img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    if not img_files:
        return
    first_idx = int(os.path.splitext(os.path.basename(img_files[0]))[0])
    offset    = first_idx % step

    for img_path in img_files:
        base = os.path.basename(img_path)
        idx  = int(os.path.splitext(base)[0])
        if idx % step != offset:
            continue
        # kopiere Bild
        dst_img = os.path.join(out_img, base)
        shutil.copy2(img_path, dst_img)
        # kopiere zugehöriges Label (.txt)
        label_src = os.path.join(lbl_dir, f"{idx:06d}.txt")
        if os.path.isfile(label_src):
            dst_lbl = os.path.join(out_lbl, f"{idx:06d}.txt")
            shutil.copy2(label_src, dst_lbl)
        label5_src = os.path.join(lbl5_dir, f"{idx:06d}.txt")
        if os.path.isfile(label5_src):
            dst_lbl5 = os.path.join(out_lbl5, f"{idx:06d}.txt")
            shutil.copy2(label5_src, dst_lbl5)

def main():
    clips = sorted(glob.glob(os.path.join(BASE_DIR, 'SNMOT-*')))
    if not clips:
        print(f"Kein Clip gefunden unter {BASE_DIR}")
        return
    for clip in clips:
        print(f"Verkleinere {clip} …")
        downsample_clip(clip, OUTPUT_DIR, STEP)
    print(f"Fertig! Neue Daten unter {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
