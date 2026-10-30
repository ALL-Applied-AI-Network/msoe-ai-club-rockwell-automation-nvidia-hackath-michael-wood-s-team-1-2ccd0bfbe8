"""Turn a stream of per-frame vial measurements into one event per filled vial.

Works online: feed frames in time order with `push()`, and it hands back a finished
`VialEvent` the moment a vial leaves the nozzle. The same class runs on a recorded
video, a folder of stills or a live camera, so what is tested offline is what runs
on the line.

A vial session opens when dyed liquid appears under the nozzle and closes when the
cart indexes away - seen as the liquid vanishing, or the level collapsing from its
peak. The vial's colour is taken from the frames at the end of the fill (level
within a few mL of the peak), and the colour of the first liquid to land is kept too:
on a recipe change that first liquid is the residue of the previous recipe, which is
the most direct evidence of carryover the camera can give.

    python src/fillevents.py --video 5            # list detected vials in a video
"""
from __future__ import annotations

import argparse, os, sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vialcolor import Measurement, linear_to_srgb

MIN_OD = 0.20          # below this the reading is glass / empty cart, not dyed liquid
MIN_CHROMA = 6.0       # an empty vial can read a false ~60 mL "level" but is colourless
MIN_LEVEL_ML = 5.0     # the ~2 mL reading is the cart in motion, not liquid
END_GAP_S = 1.5        # liquid gone this long -> the cart has indexed (gaps are 16-29 s in video 5)
DROP_ML = 25.0         # backstop only: level falls this far below the peak...
DROP_TO_ML = 18.0      # ...to an empty-vial reading...
DROP_SAMPLES = 4       # ...and stays there (the level finder glitches for seconds at a time)
SMOOTH = 3             # running median over this many samples before peak / drop logic
MIN_GAIN_ML = 12.0     # a real fill raises the level at least this much,
MIN_GAIN_OD = 0.30     # or, for pale fills the level finder cannot see, darkens this much
MIN_DURATION_S = 6.0
TAIL_FRAMES = 8        # finished colour = median of the last frames before the cart leaves,
TAIL_SKIP = 2          # skipping the last few, where the cart is already moving
TAIL_ML = 6.0          # ...and only frames this close to the peak level: at 2 s stills,
                       # 8 frames reach back into mid-fill where residue still shows
HEAD_FRAMES = 4


@dataclass
class VialEvent:
    index: int                      # 1-based count of vials seen this run
    start_s: float
    end_s: float
    start_ml: float
    final_ml: float
    absorbance: tuple               # median RGB absorbance of the finished vial
    rgb: tuple                      # rendered colour of the finished vial
    hue_deg: float
    od: float
    first_absorbance: tuple         # colour of the first liquid to land
    first_rgb: tuple
    first_hue_deg: float
    n_frames: int
    best_frame: object = field(default=None, repr=False)   # BGR image at the peak

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def _render(a) -> tuple:
    return tuple(int(v) for v in np.clip(linear_to_srgb(np.power(10.0, -np.maximum(a, 0))), 0, 255))


def _hue(rgb) -> float:
    import cv2
    lab = cv2.cvtColor(np.uint8([[rgb[::-1]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    return round(float(np.degrees(np.arctan2(lab[2] - 128, lab[1] - 128)) % 360), 1)


class FillTracker:
    def __init__(self):
        self.count = 0
        self._s: list = []          # (t, Measurement, frame, smoothed level) of the open session
        self._last_liquid_t = None
        self._below = 0             # consecutive samples under peak - DROP_ML

    @staticmethod
    def _dyed(m: Measurement) -> bool:
        return (m.present and m.liquid and m.od >= MIN_OD and m.chroma >= MIN_CHROMA
                and m.level_ml >= MIN_LEVEL_ML)

    def push(self, t: float, m: Measurement, frame=None) -> VialEvent | None:
        """Feed one measurement at time t (seconds). Returns an event when a vial closes."""
        done = None
        if self._dyed(m):
            recent = [x[1].level_ml for x in self._s[-(SMOOTH - 1):]] + [m.level_ml]
            lvl = float(np.median(recent))
            if self._s:
                peak = max(x[3] for x in self._s)
                dropped = peak - lvl > DROP_ML and lvl <= DROP_TO_ML
                self._below = self._below + 1 if dropped else 0
                if self._below >= DROP_SAMPLES:          # cart indexed with no gap seen
                    carry = self._s[-(DROP_SAMPLES - 1):] if DROP_SAMPLES > 1 else []
                    self._s = self._s[:len(self._s) - len(carry)]
                    done = self._close()
                    self._s = carry
                    self._below = 0
            self._s.append((t, m, frame, lvl))
            self._last_liquid_t = t
        elif self._s and t - self._last_liquid_t > END_GAP_S:
            done = self._close()
        return done

    def flush(self) -> VialEvent | None:
        return self._close() if self._s else None

    @property
    def filling(self) -> bool:
        return bool(self._s)

    @property
    def live_level(self) -> float:
        return self._s[-1][1].level_ml if self._s else 0.0

    def _close(self) -> VialEvent | None:
        s, self._s = self._s, []
        if not s:
            return None
        self._below = 0
        levels = np.array([x[3] for x in s])
        ods = np.array([x[1].od for x in s])
        peak = float(levels.max())
        start = float(np.median(levels[:HEAD_FRAMES]))
        gained = (peak - start >= MIN_GAIN_ML or
                  np.median(ods[-HEAD_FRAMES:]) - np.median(ods[:HEAD_FRAMES]) >= MIN_GAIN_OD)
        if not gained or s[-1][0] - s[0][0] < MIN_DURATION_S:
            return None
        tail = s[:-TAIL_SKIP][-TAIL_FRAMES:] if len(s) > TAIL_SKIP + 2 else s[-TAIL_FRAMES:]
        near = [x for x in tail if x[3] >= peak - TAIL_ML]
        tail = near or tail
        head = s[:HEAD_FRAMES]
        A = np.median([x[1].absorbance for x in tail], axis=0)
        A0 = np.median([x[1].absorbance for x in head], axis=0)
        rgb, rgb0 = _render(A), _render(A0)
        self.count += 1
        return VialEvent(index=self.count, start_s=s[0][0], end_s=s[-1][0],
                         start_ml=round(start, 1), final_ml=round(peak, 1),
                         absorbance=tuple(round(float(v), 4) for v in A), rgb=rgb,
                         hue_deg=_hue(rgb), od=round(float(A.mean()), 3),
                         first_absorbance=tuple(round(float(v), 4) for v in A0),
                         first_rgb=rgb0, first_hue_deg=_hue(rgb0), n_frames=len(s),
                         best_frame=tail[-1][2])


def events_from_video(path: str, every: int = 8, keep_frames: bool = False):
    """Run the tracker over a whole video file; returns the list of VialEvents."""
    import cv2
    from vialcolor import measure
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tr, out, i = FillTracker(), [], 0
    while True:
        if not cap.grab():
            break
        if i % every == 0:
            ok, img = cap.retrieve()
            if ok:
                ev = tr.push(i / fps, measure(img), img if keep_frames else None)
                if ev:
                    out.append(ev)
        i += 1
    ev = tr.flush()
    if ev:
        out.append(ev)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="5")
    ap.add_argument("--every", type=int, default=8, help="measure every Nth frame")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(root, "data", "vial_fill", "videos", f"vial_fill_video_{a.video}.mp4")
    evs = events_from_video(p, a.every)
    print(f"video {a.video}: {len(evs)} vials")
    print(f"{'#':>3} {'start':>7} {'end':>7} {'ml0':>4} {'mlF':>4} {'hue':>5} {'OD':>5} "
          f"{'rgb':>15} {'first hue':>9} {'first rgb':>15}")
    for e in evs:
        print(f"{e.index:>3} {e.start_s:>7.1f} {e.end_s:>7.1f} {e.start_ml:>4.0f} {e.final_ml:>4.0f} "
              f"{e.hue_deg:>5.0f} {e.od:>5.2f} {str(e.rgb):>15} {e.first_hue_deg:>9.0f} "
              f"{str(e.first_rgb):>15}")
