"""Derive the vial-fill scene geometry from the data instead of hand-guessing ROIs.

Builds a median background over video 5, then looks at where frames differ from it
and where colour (chroma) appears. Writes diagnostics to out/geom/.
"""
import csv, os
import cv2, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VF = os.path.join(ROOT, "data", "vial_fill")
OUT = os.path.join(ROOT, "out", "geom")
os.makedirs(OUT, exist_ok=True)

idx = list(csv.DictReader(open(os.path.join(VF, "image_index.csv"))))
v5 = [r["filename"] for r in idx if r["video_id"] == "5"]
print(f"video 5 frames: {len(v5)}")

imgs = []
for fn in v5[::4]:                                    # every 4th frame is plenty for a median
    im = cv2.imread(os.path.join(VF, "images", fn))
    if im is not None:
        imgs.append(im)
stack = np.stack(imgs)
bg = np.median(stack, axis=0).astype(np.uint8)
cv2.imwrite(os.path.join(OUT, "background_median.png"), bg)
print(f"median background from {len(imgs)} frames")

# Where do frames differ from the background? (cart + vials + liquid)
diff = np.abs(stack.astype(np.int16) - bg.astype(np.int16)).max(axis=3)   # per-frame max over BGR
activity = diff.mean(axis=0)
cv2.imwrite(os.path.join(OUT, "activity.png"),
            cv2.applyColorMap(cv2.normalize(activity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                              cv2.COLORMAP_INFERNO))

# Where does colour appear? LAB chroma above the background's own chroma.
lab = np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2LAB) for im in imgs]).astype(np.int16)
bglab = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).astype(np.int16)
chroma = np.sqrt(((lab[..., 1] - bglab[..., 1]) ** 2 + (lab[..., 2] - bglab[..., 2]) ** 2).astype(np.float32))
chroma_max = chroma.max(axis=0)
cv2.imwrite(os.path.join(OUT, "chroma_max.png"),
            cv2.applyColorMap(cv2.normalize(chroma_max, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                              cv2.COLORMAP_INFERNO))

# Column / row profiles of where colour shows up -> candidate vial bounds
colp = chroma_max.mean(axis=0)
rowp = chroma_max.mean(axis=1)
def spans(profile, frac=0.45):
    thr = profile.min() + frac * (profile.max() - profile.min())
    on = profile > thr
    out, s = [], None
    for i, v in enumerate(on):
        if v and s is None: s = i
        if not v and s is not None:
            if i - s > 6: out.append((s, i))
            s = None
    if s is not None and len(on) - s > 6: out.append((s, len(on)))
    return out
print("chroma column spans (x):", spans(colp))
print("chroma row spans    (y):", spans(rowp))
np.save(os.path.join(OUT, "bg.npy"), bg)
print("wrote", OUT)
