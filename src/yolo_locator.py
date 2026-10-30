"""YOLO vial locator, trained on digital-twin renders by src/train_yolo.py.

The colour measurement in src/vialcolor.py needs the vial to sit on a fixed pixel column.
That holds for the dataset camera, but a camera nudged at the factory moves the vial and
every reading goes wrong. This finds the vial under the nozzle with the detector and
reports how far it has moved from where it sits in the dataset's own reference frame,
so the dashboard can shift the frame back before measuring.

    from yolo_locator import Locator
    loc = Locator()                       # models/vial_yolo.pt
    det = loc.detect(frame)               # boxes for vial / liquid / nozzle
    dx = loc.offset_x(frame)              # coarse horizontal camera offset, or None
"""
from __future__ import annotations

import os, sys
from dataclasses import dataclass

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "models", "vial_yolo.pt")
ONNX = os.path.join(ROOT, "models", "vial_yolo.onnx")   # same network, runs on OpenCV alone
IMGSZ = 640
NMS_IOU = 0.5
CLASSES = ("vial", "liquid", "nozzle")
NOZZLE_X = (278 + 341) / 2       # vial column centre in the dataset view (vialcolor.VIAL_X)


@dataclass
class Detection:
    vial: tuple | None = None        # (x0, y0, x1, y1, conf) of the vial under the nozzle
    liquid: tuple | None = None
    nozzle: tuple | None = None

    def draw(self, img: np.ndarray) -> np.ndarray:
        # Liquid is not drawn: on real frames it fires on empty vials (twin liquid looks
        # different), so the measured level from vialcolor is what the operator sees.
        for box, col, name in ((self.vial, (0, 255, 255), "vial"), (self.nozzle, (255, 120, 0), "nozzle")):
            if box:
                x0, y0, x1, y1, c = box
                cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), col, 1)
                cv2.putText(img, f"AI {name} {c:.2f}", (int(x1) + 3, int(y0) + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        return img


def available() -> str | None:
    """Path of a model the locator can load on this machine, or None."""
    if os.path.exists(ONNX):
        return ONNX
    if os.path.exists(WEIGHTS):
        try:
            import ultralytics  # noqa: F401
            return WEIGHTS
        except ImportError:
            return None
    return None


class _OnnxYOLO:
    """YOLO11 detection head run through cv2.dnn, so no PyTorch install is needed."""

    def __init__(self, path: str):
        self.net = cv2.dnn.readNetFromONNX(path)

    def __call__(self, img: np.ndarray, conf: float):
        h, w = img.shape[:2]
        r = min(IMGSZ / h, IMGSZ / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        top, left = (IMGSZ - nh) // 2, (IMGSZ - nw) // 2      # ultralytics-style centred letterbox
        canvas = np.full((IMGSZ, IMGSZ, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        self.net.setInput(cv2.dnn.blobFromImage(canvas, 1 / 255.0, (IMGSZ, IMGSZ), swapRB=True))
        out = self.net.forward()[0].T                            # (anchors, 4 + classes)
        scores = out[:, 4:]
        cls = scores.argmax(axis=1)
        sc = scores[np.arange(len(cls)), cls]
        keep = sc >= conf
        out, cls, sc = out[keep], cls[keep], sc[keep]
        cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        boxes = np.stack([(cx - bw / 2 - left) / r, (cy - bh / 2 - top) / r,
                          (cx + bw / 2 - left) / r, (cy + bh / 2 - top) / r], axis=1)
        dets = []
        for c in np.unique(cls):
            i = np.where(cls == c)[0]
            xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes[i]]
            for j in np.array(cv2.dnn.NMSBoxes(xywh, sc[i].astype(float).tolist(), conf, NMS_IOU)).flatten():
                dets.append((int(c), float(sc[i][j]), boxes[i][j].tolist()))
        return dets


class Locator:
    def __init__(self, weights: str | None = None, conf: float = 0.25):
        weights = weights or available()
        if weights is None:
            raise FileNotFoundError("no models/vial_yolo.onnx (or vial_yolo.pt + ultralytics)")
        self.backend = "onnx" if weights.endswith(".onnx") else "ultralytics"
        if self.backend == "onnx":
            self.model = _OnnxYOLO(weights)
        else:
            from ultralytics import YOLO
            self.model = YOLO(weights)
        self.weights = weights
        self.conf = conf
        # Where the vial sits in a frame that IS aligned: the dataset's empty-vial reference.
        from vialcolor import _REF_PATH
        ref = np.load(_REF_PATH).astype(np.uint8)
        d = self.detect(ref)
        self.home = d.vial

    def detect(self, img: np.ndarray, near_x: float | None = None) -> Detection:
        if img.shape[:2] != (480, 640):
            img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
        if self.backend == "onnx":
            raw = self.model(img, self.conf)
        else:
            res = self.model(img, verbose=False, conf=self.conf)[0]
            raw = list(zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(), res.boxes.xyxy.tolist()))
        best = {c: [] for c in CLASSES}
        for c, s, b in raw:
            best[CLASSES[int(c)]].append((*b, s))
        # The vial that matters is the one under the nozzle, not a neighbour on the track.
        # Its home column is a steadier anchor than the nozzle box, which YOLO misses often.
        home = getattr(self, "home", None)
        x = near_x if near_x is not None else ((home[0] + home[2]) / 2 if home else NOZZLE_X)
        pick = lambda bs: min(bs, key=lambda b: abs((b[0] + b[2]) / 2 - x)) if bs else None
        return Detection(vial=pick(best["vial"]), liquid=pick(best["liquid"]),
                         nozzle=max(best["nozzle"], key=lambda b: b[4]) if best["nozzle"] else None)

    def offset_x(self, img: np.ndarray) -> int | None:
        """Coarse horizontal shift of the vial from its home column, or None if not found.

        Only x: on real frames the box's top and bottom edges wander by tens of pixels,
        while its centre column holds to ~2 px. The dashboard refines x and finds y by
        template matching around this estimate, which lands within 1 px on shifts up to
        90 px (template matching alone loses the vial past ~60 px).
        """
        if self.home is None:
            return None
        v = self.detect(img).vial
        if v is None:
            return None
        return int(round((v[0] + v[2]) / 2 - (self.home[0] + self.home[2]) / 2))
