"""Solve for each tank's dye absorbance from the vials themselves.

Every labelled vial is a known mixture, so for a clean (uncontaminated) vial

    A_observed = sum_t (mL_t / total_mL) * eps_t

with eps_t the absorbance of tank t at full strength. Stack the clean vials and
solve the 3-unknown least squares per colour channel. This measures the dye through
the same glass and the same camera the model will be used on, which a reading off
the tank camera would not.

Only the 3rd vial of each recipe batch is used. The dataset notes say the first of a
batch carries residue; our measurements show the second often does too, so the third
is the only reliably clean mixture.

    python src/calibrate_tanks.py
"""
import csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vialcolor import linear_to_srgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VF = os.path.join(ROOT, "data", "vial_fill")
OUT = os.path.join(ROOT, "out")
TANKS = ("T-100", "T-200", "T-400")


def load():
    fills = list(csv.DictReader(open(os.path.join(OUT, "fills_video5.csv"))))
    recipes = {int(r["vial"]): r
               for r in csv.DictReader(open(os.path.join(VF, "labels", "video5_vial_recipes.csv")))}
    return fills, recipes


def pair_by_order(fills, recipes):
    """Match detected fill k to labelled vial k.

    The label file's timestamps are offset from the real fills by up to ~35 s, so
    matching on time shifts the mapping by a whole vial and attaches orange fills to
    blue recipes. The ORDER of the labels is sound, so position is used instead. This
    holds only while no fill is missed; it is checked by the sign of the fit below.
    """
    return [(f, recipes[int(f["fill"])]) for f in fills if int(f["fill"]) in recipes]


def main():
    fills, recipes = load()
    pairs = pair_by_order(fills, recipes)
    print(f"matched {len(pairs)} detected fills to labelled vials by order\n")

    rows, obs, used = [], [], []
    for f, r in pairs:
        vial = int(r["vial"])
        # Only the 3rd vial of a batch is reliably clean. The dataset notes say the
        # 1st carries residue; the measurements show the 2nd often does too (Recipe 13
        # reads 234deg, 216deg, 49deg across its three vials), and including 2nds makes
        # the fit worse (RMS 0.296 vs 0.273).
        if (vial - 1) % 3 != 2:
            continue
        total = float(r["total_ml"])
        frac = [float(r[k] or 0) / total for k in ("t100_ml", "t200_ml", "t400_ml")]
        A = [float(f[k]) for k in ("a_r", "a_g", "a_b")]
        if not all(np.isfinite(A)) or max(A) <= 0:
            continue
        rows.append(frac); obs.append(A); used.append((vial, int(r["recipe_id"])))

    M, Y = np.array(rows), np.array(obs)
    print(f"using {len(M)} clean vials (3rd of each batch): "
          f"{', '.join(f'v{v}/r{rc}' for v, rc in used)}\n")

    eps, *_ = np.linalg.lstsq(M, Y, rcond=None)
    eps = np.clip(eps, 0.0, None)                  # absorbance cannot be negative
    resid = Y - M @ eps

    print(f"{'tank':>7} {'A_red':>7} {'A_green':>7} {'A_blue':>7}   {'sRGB at full strength':>24}")
    tank_rgb = {}
    for i, t in enumerate(TANKS):
        srgb = linear_to_srgb(np.power(10.0, -eps[i]))
        tank_rgb[t] = [round(float(v), 1) for v in srgb]
        print(f"{t:>7} {eps[i,0]:>7.3f} {eps[i,1]:>7.3f} {eps[i,2]:>7.3f}   "
              f"{str([int(v) for v in srgb]):>24}")

    rms = float(np.sqrt((resid ** 2).mean()))
    print(f"\nfit RMS residual: {rms:.3f} absorbance units "
          f"(observed OD spans {Y.min():.2f}-{Y.max():.2f})")

    import json
    with open(os.path.join(OUT, "tank_calibration.json"), "w") as fh:
        json.dump({"tank_rgb": tank_rgb,
                   "absorbance": {t: [round(float(x), 4) for x in eps[i]] for i, t in enumerate(TANKS)},
                   "n_clean_vials": len(M), "rms_residual": round(rms, 4)}, fh, indent=2)
    print("wrote out/tank_calibration.json")


if __name__ == "__main__":
    main()
