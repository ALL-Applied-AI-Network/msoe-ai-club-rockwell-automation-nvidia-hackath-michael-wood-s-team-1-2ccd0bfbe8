"""Measure the liquid in the fill-station vial from a `vial_fill` camera frame.

    from vialcolor import measure
    m = measure("data/vial_fill/images/vial_fill_20260923_195853_579357.jpg")
    print(m.rgb, m.hue_deg, m.level_ml)

    python src/vialcolor.py <image> --viz out/check.png

Colour is measured as ABSORBANCE against an empty-vial reference, not as raw pixels.
Pixels are linearised out of sRGB first: Beer-Lambert is only valid on linear light.
The vial sits against white paper at the top of the frame and dark steel lower down,
so a raw reading of the same liquid changes with fill height. Beer-Lambert absorbance
A = -log10(I / I0) cancels the backdrop, the glass and the illumination, leaving a
property of the dye alone. `rgb` is then rendered as the colour the liquid would show
against a white background, which is comparable across heights and across frames.

No fitted parameters: every constant is scene geometry (from src/explore_geometry.py)
or a fixed colourimetric rule, and the reference image is a median of empty-vial
frames. Nothing is trained on the real images - this is an instrument, not a model.
"""
from __future__ import annotations

import argparse, json, os
from dataclasses import dataclass, asdict, field

import cv2
import numpy as np

# --- scene geometry, derived by src/explore_geometry.py on video 5 -----------
VIAL_X = (278, 341)              # outer glass walls of the vial under the nozzle
CORE_X = (293, 327)              # central strip, clear of the walls' specular edges
VIAL_Y = (266, 422)              # usable glass column: below the holder collar, above the base
COLLAR = (205, 262, 268, 392)    # y0, y1, x0, x1 - black holder block

COLLAR_L_MAX = 120.0             # above this the collar band is bare backdrop: no cart
MIN_STEP = 0.20                 # smallest air->liquid absorbance step that counts
MIN_RUN = 4                      # rows the liquid region must span
I0_MIN = 26.0                    # reference rows darker than this are too noisy to use
VIAL_CAPACITY_ML = 80.0          # stated operating limit; the column is scaled to it

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """sRGB 0-255 -> linear light 0-1. Beer-Lambert is only valid on linear light."""
    x = np.asarray(x, np.float64) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, np.float64), 0.0, 1.0)
    return 255.0 * np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


_REF_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "out", "empty_vial_reference.npy")
_REF = None


def _reference() -> np.ndarray:
    global _REF
    if _REF is None:
        if not os.path.exists(_REF_PATH):
            raise FileNotFoundError(
                f"missing {_REF_PATH} - run `python src/build_reference.py` first")
        _REF = np.load(_REF_PATH).astype(np.float32)
    return _REF


@dataclass
class Measurement:
    source: str
    present: bool                 # is a vial parked under the nozzle
    liquid: bool                  # is there measurable liquid in it
    confidence: str               # high | low | none
    level_px: float = 0.0
    level_frac: float = 0.0       # 0-1 of the usable glass column
    level_ml: float = 0.0         # level_frac scaled to the 80 mL working volume
    rgb: tuple = (255, 255, 255)  # colour the liquid would show against white
    hsv: tuple = (0, 0, 0)
    lab: tuple = (100.0, 0.0, 0.0)
    hue_deg: float = 0.0
    chroma: float = 0.0           # LAB chroma of the transmitted colour
    absorbance: tuple = (0.0, 0.0, 0.0)   # mean A in R, G, B - the dye signature
    od: float = 0.0               # mean optical density over the three channels
    liquid_top_y: int = 0
    profile: list = field(default_factory=list, repr=False)   # [y, A_r, A_g, A_b] per row


def _absorbance_profile(img: np.ndarray):
    """Per-row RGB absorbance down the centre of the vial, plus a usability mask."""
    y0, y1 = VIAL_Y
    x0, x1 = CORE_X
    cur = img[y0:y1, x0:x1].astype(np.float32)[..., ::-1]          # -> RGB
    ref = _reference()[y0:y1, x0:x1][..., ::-1]

    # Reject per-row speculars in both current and reference before the ratio.
    lum_c, lum_r = cur.mean(axis=2), ref.mean(axis=2)
    keep = (lum_c < np.percentile(lum_c, 90, axis=1, keepdims=True)) & \
           (lum_r < np.percentile(lum_r, 90, axis=1, keepdims=True))

    rows_c = np.zeros((cur.shape[0], 3), np.float32)
    rows_r = np.zeros((cur.shape[0], 3), np.float32)
    for i in range(cur.shape[0]):
        k = keep[i] if keep[i].sum() >= 6 else np.ones(cur.shape[1], bool)
        rows_c[i] = np.median(cur[i][k], axis=0)
        rows_r[i] = np.median(ref[i][k], axis=0)

    usable = rows_r.mean(axis=1) >= I0_MIN
    lin_c = np.clip(srgb_to_linear(rows_c), 1e-5, None)
    lin_r = np.clip(srgb_to_linear(rows_r), 1e-5, None)
    A = -np.log10(lin_c / lin_r)
    return np.clip(A, -0.5, 4.0).astype(np.float32), usable, rows_r


def _find_liquid_top(A: np.ndarray, usable: np.ndarray) -> tuple[int, str]:
    """Split the column into air above and liquid below at the sharpest step.

    The glass itself has a small, height-dependent absorbance baseline, so a fixed
    threshold either clips pale fills or runs away up the empty column. The liquid
    surface is instead the single row that best separates the column into a low
    region above and a high region below - a 1-D two-class split, which needs no
    threshold on the absorbance itself, only on the size of the step.
    """
    a = A.mean(axis=1).copy()
    n = len(a)
    a[~usable] = np.nan
    if np.isnan(a).all():
        return -1, "none"
    filled = np.where(np.isnan(a), np.nanmedian(a), a)

    best_i, best_step = -1, 0.0
    for i in range(MIN_RUN, n - MIN_RUN + 1):
        step = filled[i:].mean() - filled[:i].mean()
        if step > best_step:
            best_i, best_step = i, step

    # A column that is uniformly dark has no step but is still full of liquid.
    if best_step < MIN_STEP:
        if filled.mean() > MIN_STEP * 2:
            return 0, "low"
        return -1, "none"
    return best_i, ("high" if best_step >= MIN_STEP * 2 else "low")


def measure(src) -> Measurement:
    img = cv2.imread(src) if isinstance(src, str) else src
    if img is None:
        raise FileNotFoundError(f"cannot read image: {src}")
    if img.shape[:2] != (480, 640):
        raise ValueError(f"expected a 640x480 vial_fill frame, got {img.shape[1]}x{img.shape[0]}")
    name = src if isinstance(src, str) else "<array>"

    cy0, cy1, cx0, cx1 = COLLAR
    collar_L = cv2.cvtColor(img[cy0:cy1, cx0:cx1], cv2.COLOR_BGR2LAB)[..., 0].mean() * 100.0 / 255.0
    if collar_L > COLLAR_L_MAX:
        return Measurement(source=name, present=False, liquid=False, confidence="none")

    A, usable, _ = _absorbance_profile(img)
    top, conf = _find_liquid_top(A, usable)
    m = Measurement(source=name, present=True, liquid=top >= 0, confidence=conf,
                    profile=[[int(VIAL_Y[0] + i), *np.round(a, 3).tolist()] for i, a in enumerate(A)])
    if top < 0:
        return m

    height = len(A)
    m.liquid_top_y = int(VIAL_Y[0] + top)
    m.level_px = float(height - top)
    m.level_frac = float(m.level_px / height)
    m.level_ml = round(m.level_frac * VIAL_CAPACITY_ML, 1)

    # Average absorbance over the body of the liquid, using only rows with a
    # bright enough reference to give a trustworthy ratio.
    lo = min(top + 3, height - 1)
    sel = np.zeros(height, bool)
    sel[lo:] = True
    sel &= usable
    if sel.sum() < 3:
        m.confidence = "low"
        sel[lo:] = True
    Amean = A[sel].mean(axis=0)
    m.absorbance = tuple(round(float(v), 3) for v in Amean)
    m.od = round(float(Amean.mean()), 3)

    # Render the dye as it would look transmitted through white light.
    rgb = np.clip(linear_to_srgb(np.power(10.0, -np.maximum(Amean, 0.0))), 0, 255)
    px = np.uint8([[rgb[::-1]]])                                  # -> BGR for OpenCV
    lab = cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    lab = np.array([lab[0] * 100.0 / 255.0, lab[1] - 128.0, lab[2] - 128.0])
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)

    m.rgb = tuple(int(v) for v in rgb)
    m.lab = tuple(round(float(v), 2) for v in lab)
    m.hsv = (round(float(hsv[0]) * 2, 1), round(float(hsv[1]) / 2.55, 1), round(float(hsv[2]) / 2.55, 1))
    m.chroma = round(float(np.hypot(lab[1], lab[2])), 2)
    m.hue_deg = round(float(np.degrees(np.arctan2(lab[2], lab[1])) % 360), 1)
    return m


def visualise(src: str, out_path: str) -> Measurement:
    img = cv2.imread(src)
    m = measure(img)
    vis = img.copy()
    cv2.rectangle(vis, (CORE_X[0], VIAL_Y[0]), (CORE_X[1], VIAL_Y[1]), (200, 200, 200), 1)
    if m.liquid:
        cv2.line(vis, (VIAL_X[0] - 8, m.liquid_top_y), (VIAL_X[1] + 8, m.liquid_top_y), (0, 255, 255), 2)
        sw = tuple(int(c) for c in m.rgb[::-1])
        cv2.rectangle(vis, (VIAL_X[1] + 16, VIAL_Y[0]), (VIAL_X[1] + 60, VIAL_Y[0] + 44), sw, -1)
        cv2.rectangle(vis, (VIAL_X[1] + 16, VIAL_Y[0]), (VIAL_X[1] + 60, VIAL_Y[0] + 44), (40, 40, 40), 1)
        txt = f"{m.level_ml:.0f}mL  hue{m.hue_deg:.0f}  OD{m.od:.2f}"
    else:
        txt = "no vial" if not m.present else "empty"
    for col, th in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(vis, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, th)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, vis)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Measure the liquid colour in a vial_fill frame.")
    ap.add_argument("image")
    ap.add_argument("--viz", help="write an annotated copy here")
    ap.add_argument("--json", action="store_true", help="print the full record as JSON")
    a = ap.parse_args()
    m = visualise(a.image, a.viz) if a.viz else measure(a.image)
    if a.json:
        d = asdict(m); d.pop("profile")
        print(json.dumps(d, indent=2))
    elif not m.present:
        print("no vial parked under the nozzle")
    elif not m.liquid:
        print("vial present, no liquid detected")
    else:
        print(f"RGB {m.rgb}   HSV {m.hsv}   LAB {m.lab}")
        print(f"absorbance R/G/B {m.absorbance}  OD {m.od}")
        print(f"hue {m.hue_deg}deg  chroma {m.chroma}  level {m.level_ml} mL "
              f"({m.level_frac*100:.0f}% of column)  confidence={m.confidence}")
