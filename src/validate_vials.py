"""Run vialcolor over every labelled fill window and compare with the recipe table."""
import csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vialcolor import measure

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VF = os.path.join(ROOT, "data", "vial_fill")
IMG = os.path.join(VF, "images")

frames = list(csv.DictReader(open(os.path.join(VF, "labels", "video5_frame_labels.csv"))))
recipes = {int(r["vial"]): r for r in csv.DictReader(open(os.path.join(VF, "labels", "video5_vial_recipes.csv")))}

by_vial = {}
for r in frames:
    if r["in_fill"] == "1":
        by_vial.setdefault(int(r["vial"]), []).append(r["filename"])

def ml(v): return float(v) if v else 0.0

print(f"{'vial':>4} {'rcp':>3} {'T100':>4} {'T200':>4} {'T400':>4} {'lbl_mL':>6} "
      f"{'obs_mL':>6} {'hue':>6} {'chr':>5} {'RGB':>14}  {'conf':>4}  pos")
print("-" * 86)

rows = []
for v in sorted(by_vial):
    rec = recipes[v]
    best = None
    for fn in by_vial[v]:
        p = os.path.join(IMG, fn)
        if not os.path.exists(p):
            continue
        m = measure(p)
        if m.present and m.liquid and (best is None or m.level_ml > best.level_ml):
            best = m
    if best is None:
        print(f"{v:>4} {rec['recipe_id']:>3}   --- no usable frame ---"); continue
    pos = ["1st", "2nd", "3rd"][(v - 1) % 3]
    print(f"{v:>4} {rec['recipe_id']:>3} {ml(rec['t100_ml']):>4.0f} {ml(rec['t200_ml']):>4.0f} "
          f"{ml(rec['t400_ml']):>4.0f} {ml(rec['total_ml']):>6.0f} {best.level_ml:>6.1f} "
          f"{best.hue_deg:>6.1f} {best.chroma:>5.1f} {str(best.rgb):>14}  {best.confidence:>4}  {pos}")
    rows.append((v, int(rec["recipe_id"]), ml(rec["total_ml"]), best))

obs = np.array([r[3].level_ml for r in rows]); lbl = np.array([r[2] for r in rows])
print("-" * 86)
print(f"n={len(rows)}  level MAE={np.abs(obs-lbl).mean():.1f} mL   "
      f"corr(obs,label)={np.corrcoef(obs,lbl)[0,1]:.3f}")
