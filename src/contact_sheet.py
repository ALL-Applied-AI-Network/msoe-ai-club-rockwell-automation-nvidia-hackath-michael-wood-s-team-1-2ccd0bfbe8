"""Tile the peak frame of every detected fill, with its measured colour, as one sheet."""
import csv, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vialcolor import VIAL_X, VIAL_Y

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VF = os.path.join(ROOT, "data", "vial_fill")
video = sys.argv[1] if len(sys.argv) > 1 else "5"
fills = list(csv.DictReader(open(os.path.join(ROOT, "out", f"fills_video{video}.csv"))))

PAD, SWATCH, COLS = 6, 26, 10
tiles = []
for f in fills:
    im = cv2.imread(os.path.join(VF, "images", f["peak_frame"]))
    crop = im[VIAL_Y[0] - 16:VIAL_Y[1] + 8, VIAL_X[0] - 10:VIAL_X[1] + 10]
    crop = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
    h, w = crop.shape[:2]
    tile = np.full((h + SWATCH + 22, w, 3), 245, np.uint8)
    tile[:h] = crop
    bgr = (int(f["b"]), int(f["g"]), int(f["r"]))
    cv2.rectangle(tile, (0, h), (w, h + SWATCH), bgr, -1)
    cv2.putText(tile, f"#{f['fill']} {float(f['final_ml']):.0f}mL", (3, h + SWATCH + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1)
    tiles.append(tile)

H = max(t.shape[0] for t in tiles); W = max(t.shape[1] for t in tiles)
rows = []
for i in range(0, len(tiles), COLS):
    chunk = tiles[i:i + COLS]
    row = np.full((H, W * COLS + PAD * (COLS - 1), 3), 245, np.uint8)
    for j, t in enumerate(chunk):
        row[:t.shape[0], j * (W + PAD):j * (W + PAD) + t.shape[1]] = t
    rows.append(row)
sheet = np.full((sum(r.shape[0] for r in rows) + PAD * (len(rows) - 1), rows[0].shape[1], 3), 245, np.uint8)
y = 0
for r in rows:
    sheet[y:y + r.shape[0]] = r; y += r.shape[0] + PAD
out = os.path.join(ROOT, "out", f"contact_sheet_video{video}.png")
cv2.imwrite(out, sheet)
print(f"{len(tiles)} fills -> {os.path.relpath(out, ROOT)}  ({sheet.shape[1]}x{sheet.shape[0]})")
