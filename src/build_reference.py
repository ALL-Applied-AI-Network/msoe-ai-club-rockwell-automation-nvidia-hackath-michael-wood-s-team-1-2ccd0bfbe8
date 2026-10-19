"""Build the empty-vial reference image I0 used as the Beer-Lambert baseline.

The vial sits against white paper at the top of the frame and dark steel lower down,
so raw pixel colour depends on fill height. Dividing by an empty-vial reference
cancels the backdrop, the glass and the lighting, leaving absorbance - a property of
the dye alone.

The empty frames are found from the images themselves, NOT from the label file: the
labelled fill windows are offset from the real fills, so
"first frame of a fill" is not reliably an empty vial. An empty column is simply the
brightest one, since any dye only ever absorbs light.
"""
import csv, os
import cv2, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VF = os.path.join(ROOT, "data", "vial_fill")
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

CORE_X, VIAL_Y = (293, 327), (266, 422)
COLLAR = (205, 262, 268, 392)
COLLAR_L_MAX = 120.0
BRIGHTEST_FRAC = 0.12

idx = list(csv.DictReader(open(os.path.join(VF, "image_index.csv"))))
cands = []
for r in (r for r in idx if r["video_id"] == "5"):
    im = cv2.imread(os.path.join(VF, "images", r["filename"]))
    if im is None:
        continue
    y0, y1, x0, x1 = COLLAR
    if cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2LAB)[..., 0].mean() * 100 / 255 > COLLAR_L_MAX:
        continue                                   # no cart parked under the nozzle
    col = im[VIAL_Y[0]:VIAL_Y[1], CORE_X[0]:CORE_X[1]]
    # Score on the lower half, where liquid always sits first.
    cands.append((float(np.median(col[col.shape[0] // 2:].mean(axis=2))), im))

cands.sort(key=lambda t: -t[0])
keep = cands[: max(8, int(len(cands) * BRIGHTEST_FRAC))]
ref = np.median(np.stack([im for _, im in keep]), axis=0).astype(np.uint8)
np.save(os.path.join(OUT, "empty_vial_reference.npy"), ref)
cv2.imwrite(os.path.join(OUT, "empty_vial_reference.png"), ref)
print(f"{len(cands)} parked-vial frames; reference = median of the brightest {len(keep)}")
print(f"  brightness: kept {keep[-1][0]:.1f}-{keep[0][0]:.1f}, rejected down to {cands[-1][0]:.1f}")
print("wrote out/empty_vial_reference.npy")
