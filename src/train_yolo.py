"""Train a YOLO vial / liquid / nozzle detector on digital-twin renders only; test on real frames.

    python src/train_yolo.py                       # train on every set in data/twin_yolo/, then evaluate
    python src/train_yolo.py --epochs 80 --imgsz 640
    python src/train_yolo.py --eval-only           # re-score models/vial_yolo.pt on real frames

Training data comes from "Get Training Data.bat" (Digital Twin\\scripts\\server\\synth_dataset.py):
random fill moment, liquid colour and lighting, with exact boxes from Replicator. No real
image is used for training, as the event guide asks.

The real-frame test has no hand-drawn boxes, so it is scored against what the measurement
instrument already knows: the vial's column (src/vialcolor.py geometry) and, when there is
liquid, the liquid surface it finds. It reports how often the detector finds the vial, how
far off its centre is, and whether it agrees on liquid present / absent.
"""
from __future__ import annotations

import argparse, csv, glob, json, os, random, shutil, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vialcolor import measure, VIAL_X, VIAL_Y

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TWIN_SETS = os.path.join(ROOT, "data", "twin_yolo")
WORK = os.path.join(ROOT, "out", "yolo")
WEIGHTS = os.path.join(ROOT, "models", "vial_yolo.pt")
BASE = os.path.join(ROOT, "models", "yolo11n.pt")
CLASSES = ["vial", "liquid", "nozzle"]
VAL_FRAC = 0.15
MIN_OD = 0.20                    # same "dyed liquid" bar the fill tracker uses


def build_dataset() -> str:
    imgs = sorted(glob.glob(os.path.join(TWIN_SETS, "*", "images", "*.png")))
    if not imgs:
        sys.exit(f"no twin renders in {TWIN_SETS} - run 'Get Training Data.bat' first")
    random.Random(0).shuffle(imgs)
    n_val = max(1, int(len(imgs) * VAL_FRAC))
    for split, part in (("val", imgs[:n_val]), ("train", imgs[n_val:])):
        for sub in ("images", "labels"):
            d = os.path.join(WORK, "dataset", sub, split)
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d)
        for p in part:
            name = os.path.splitext(os.path.basename(p))[0]
            lbl = os.path.join(os.path.dirname(os.path.dirname(p)), "labels", name + ".txt")
            shutil.copy(p, os.path.join(WORK, "dataset", "images", split))
            if os.path.exists(lbl):
                shutil.copy(lbl, os.path.join(WORK, "dataset", "labels", split))
    yaml = os.path.join(WORK, "twin.yaml")
    with open(yaml, "w") as f:
        f.write(f"path: {os.path.join(WORK, 'dataset').replace(os.sep, '/')}\n"
                "train: images/train\nval: images/val\n"
                "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    print(f"twin dataset: {len(imgs) - n_val} train / {n_val} val images")
    return yaml


def train(yaml: str, epochs: int, imgsz: int, batch: int):
    from ultralytics import YOLO
    model = YOLO(BASE if os.path.exists(BASE) else "yolo11n.pt")
    # Geometric and photometric augmentation stands in for the camera offset and lighting
    # the real line will have; the twin itself already varies colour and light level.
    model.train(data=yaml, epochs=epochs, imgsz=imgsz, batch=batch, device="cpu",
                project=WORK, name="train", exist_ok=True, workers=2, patience=20,
                hsv_h=0.05, hsv_s=0.5, hsv_v=0.4, translate=0.15, scale=0.35,
                degrees=2.0, fliplr=0.5, mosaic=1.0, verbose=False, plots=True)
    best = os.path.join(WORK, "train", "weights", "best.pt")
    shutil.copy(best, WEIGHTS)
    print(f"weights -> {os.path.relpath(WEIGHTS, ROOT)}")


def evaluate(n: int, conf: float) -> dict:
    """Score the detector on real vial_fill stills it never saw."""
    import cv2
    from ultralytics import YOLO
    model = YOLO(WEIGHTS)
    idx = [r for r in csv.DictReader(open(os.path.join(ROOT, "data", "vial_fill", "image_index.csv")))]
    random.Random(1).shuffle(idx)
    cx_true = (VIAL_X[0] + VIAL_X[1]) / 2
    found = present = 0
    cx_err, top_err = [], []
    liq = dict(tp=0, fp=0, fn=0, tn=0)
    rows = []
    for r in idx[:n]:
        p = os.path.join(ROOT, "data", "vial_fill", "images", r["filename"])
        img = cv2.imread(p)
        if img is None:
            continue
        m = measure(img)
        res = model(img, verbose=False, conf=conf)[0]
        boxes = {c: [] for c in CLASSES}
        for c, s, b in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(), res.boxes.xyxy.tolist()):
            boxes[CLASSES[int(c)]].append((s, b))
        # The vial under the nozzle is the one whose centre is nearest the nozzle column.
        vial = min(boxes["vial"], key=lambda sb: abs((sb[1][0] + sb[1][2]) / 2 - cx_true), default=None)
        if m.present:
            present += 1
            if vial and abs((vial[1][0] + vial[1][2]) / 2 - cx_true) < (VIAL_X[1] - VIAL_X[0]):
                found += 1
                cx_err.append(abs((vial[1][0] + vial[1][2]) / 2 - cx_true))
        truth_liquid = m.present and m.liquid and m.od >= MIN_OD
        pred = [b for _, b in boxes["liquid"] if VIAL_X[0] - 20 < (b[0] + b[2]) / 2 < VIAL_X[1] + 20]
        k = ("tp" if pred else "fn") if truth_liquid else ("fp" if pred else "tn")
        liq[k] += 1
        if truth_liquid and pred:
            top_err.append(abs(min(b[1] for b in pred) - m.liquid_top_y))
        rows.append(dict(file=r["filename"], present=int(m.present), liquid=int(truth_liquid),
                         vial_found=int(vial is not None), liquid_pred=int(bool(pred))))
    nl = sum(liq.values())
    rep = dict(frames=len(rows), vial_present=present,
               vial_recall=round(found / present, 3) if present else None,
               vial_centre_err_px=round(float(np.median(cx_err)), 1) if cx_err else None,
               liquid=liq, liquid_accuracy=round((liq["tp"] + liq["tn"]) / nl, 3) if nl else None,
               liquid_top_err_px=round(float(np.median(top_err)), 1) if top_err else None)
    os.makedirs(WORK, exist_ok=True)
    with open(os.path.join(WORK, "real_eval.json"), "w") as f:
        json.dump(dict(report=rep, rows=rows), f, indent=1)
    print("\nreal-frame test (trained on twin only):")
    for k, v in rep.items():
        print(f"  {k:<20} {v}")
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--eval-frames", type=int, default=300)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    if not a.eval_only:
        train(build_dataset(), a.epochs, a.imgsz, a.batch)
    evaluate(a.eval_frames, a.conf)


if __name__ == "__main__":
    main()
