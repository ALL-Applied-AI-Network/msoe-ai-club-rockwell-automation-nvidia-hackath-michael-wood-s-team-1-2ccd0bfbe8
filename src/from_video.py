"""Pull individual vial images straight out of an .mp4 at full 30 fps.

The sampled stills are only every ~2 s, which is too coarse to see a fill develop.
The videos are 30 fps, so this gives 60x the temporal resolution for the same scene.
`measure()` works on video frames unchanged - the camera and geometry are identical.

    python src/from_video.py --video 5 --every 15            # measure every 0.5 s
    python src/from_video.py --video 5 --every 15 --crops out/crops_v5
"""
import argparse, csv, os, sys
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vialcolor import measure, VIAL_X, VIAL_Y

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--video", default="5")
ap.add_argument("--every", type=int, default=15, help="sample every Nth frame (30 = 1 s)")
ap.add_argument("--crops", help="also write a cropped vial image per sampled frame")
a = ap.parse_args()

path = os.path.join(ROOT, "data", "vial_fill", "videos", f"vial_fill_video_{a.video}.mp4")
cap = cv2.VideoCapture(path)
if not cap.isOpened():
    sys.exit(f"cannot open {path}")
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if a.crops:
    os.makedirs(a.crops, exist_ok=True)

out = os.path.join(ROOT, "out", f"video_frames_video{a.video}.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
rows, n = [], 0
for fr in range(0, total, a.every):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
    ok, img = cap.read()
    if not ok:
        continue
    m = measure(img)
    rows.append(dict(frame=fr, video_s=round(fr / fps, 3), present=int(m.present),
                     liquid=int(m.liquid), level_ml=m.level_ml, od=m.od,
                     hue_deg=m.hue_deg, chroma=m.chroma,
                     r=m.rgb[0], g=m.rgb[1], b=m.rgb[2], confidence=m.confidence))
    if a.crops and m.liquid:
        crop = img[VIAL_Y[0] - 16:VIAL_Y[1] + 8, VIAL_X[0] - 10:VIAL_X[1] + 10]
        cv2.imwrite(os.path.join(a.crops, f"vial_f{fr:06d}.png"), crop)
        n += 1

with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"video {a.video}: {total} frames @ {fps:.0f} fps, sampled {len(rows)} every {a.every} "
      f"({a.every / fps:.2f} s)")
print(f"  -> {os.path.relpath(out, ROOT)}" + (f"   crops: {n} -> {a.crops}" if a.crops else ""))
