"""Recipe -> expected colour, carryover prediction, and the per-vial verdict.

This is the reasoning half of the monitor. Given the recipe a vial was SUPPOSED to
get and the recipes that ran before it, it predicts

    clean     what the vial should look like (Beer-Lambert blend of the tanks)
    carryover what it will look like once residue in the shared mixing path is
              included (the hydraulic delay line from src/carryover.py)

and then compares both with what the camera measured. If the vial is off-colour and
the carryover prediction explains it, the cause is named; if not, the operator is
told to look at the tanks / dosing instead. Distances are in camera absorbance
(RGB optical density), the same space src/vialcolor.py measures in.

The fitted numbers (tank dye coefficients, hold-up volume, thresholds) live in a
JSON file written by src/calibrate_model.py, so the frozen model that runs on the
line is a file you can diff, not a retrain.
"""
from __future__ import annotations

import json, os, sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from carryover import TANKS, DISPENSE_ORDER, simulate, design
from vialcolor import linear_to_srgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(ROOT, "models", "color_model.json")

# Station operating limits from the event guide; recipes outside them are refused.
MAX_TOTAL_ML = 80.0
MAX_T200_ML = 50.0
MIN_COMPONENT_ML = 10.0


@dataclass
class Recipe:
    name: str
    t100: float = 0.0
    t200: float = 0.0
    t400: float = 0.0

    @property
    def vols(self) -> dict:
        return {"T-100": self.t100, "T-200": self.t200, "T-400": self.t400}

    @property
    def total(self) -> float:
        return self.t100 + self.t200 + self.t400

    def item(self) -> dict:
        """The record format the carryover simulator consumes."""
        return dict(vols=self.vols, total=self.total)

    def problems(self) -> list:
        p = []
        if self.total <= 0:
            p.append("recipe dispenses nothing")
        if self.total > MAX_TOTAL_ML:
            p.append(f"total {self.total:.0f} mL exceeds the {MAX_TOTAL_ML:.0f} mL vial limit")
        if self.t200 > MAX_T200_ML:
            p.append(f"T-200 {self.t200:.0f} mL exceeds the {MAX_T200_ML:.0f} mL fault limit")
        for t, v in self.vols.items():
            if 0 < v < MIN_COMPONENT_ML:
                p.append(f"{t} {v:.0f} mL is under the {MIN_COMPONENT_ML:.0f} mL practical minimum")
        return p

    def describe(self) -> str:
        order = [t for t in DISPENSE_ORDER if self.vols[t] > 0]
        return ", ".join(f"{t} {self.vols[t]:.0f} mL" for t in order)

    @staticmethod
    def from_dict(d: dict) -> "Recipe":
        g = lambda *ks: float(next((d[k] for k in ks if d.get(k) not in (None, "")), 0) or 0)
        return Recipe(name=str(d.get("name") or d.get("recipe_id") or "?"),
                      t100=g("t100", "t100_ml", "T-100"), t200=g("t200", "t200_ml", "T-200"),
                      t400=g("t400", "t400_ml", "T-400"))


def to_rgb(a) -> tuple:
    return tuple(int(v) for v in np.clip(linear_to_srgb(np.power(10.0, -np.maximum(np.asarray(a, float), 0))), 0, 255))


def hexcolor(a) -> str:
    return "#%02x%02x%02x" % to_rgb(a)


class ColorModel:
    def __init__(self, params: dict):
        self.p = params
        self.coef = np.array([params["baseline"], params["T-200"], params["T-400"]], float)
        self.hold_up = float(params["hold_up_ml"])
        self.well_mixed = params.get("hydraulics", "well-mixed") == "well-mixed"
        self.tol = float(params.get("tolerance", 0.25))           # clean vs observed
        self.explain_gain = float(params.get("explain_gain", 0.30))

    @staticmethod
    def load(path: str = DEFAULT_MODEL) -> "ColorModel":
        with open(path) as f:
            return ColorModel(json.load(f))

    # ------------------------------------------------------------- prediction
    def clean(self, r: Recipe) -> np.ndarray:
        f = np.array([[r.vols[t] / r.total for t in TANKS]])
        return (design(f) @ self.coef)[0]

    def with_history(self, history: list, r: Recipe) -> np.ndarray:
        """Predicted colour of the NEXT vial of `r`, given the vials already dispensed."""
        seq = [h.item() for h in history] + [r.item()]
        if len(seq) == 1:
            return self.clean(r)
        mix = simulate(seq, self.hold_up, self.well_mixed)
        return (design(mix[-1:]) @ self.coef)[0]

    def transition_risk(self, prev: Recipe, nxt: Recipe, prev_count: int = 3) -> dict:
        """How far off the first vial of `nxt` will be after `prev_count` vials of `prev`."""
        dirty = self.with_history([prev] * prev_count, nxt)
        clean = self.clean(nxt)
        err = float(np.linalg.norm(dirty - clean))
        # How many vials until the residue has flushed below tolerance.
        n, hist = 1, [prev] * prev_count
        while n < 6:
            hist = hist + [nxt]
            if np.linalg.norm(self.with_history(hist, nxt) - clean) < self.tol:
                break
            n += 1
        level = "HIGH" if err > 2 * self.tol else "MEDIUM" if err > self.tol else "LOW"
        return dict(prev=prev.name, next=nxt.name, error=round(err, 3), level=level,
                    first_vial_hex=hexcolor(dirty), clean_hex=hexcolor(clean),
                    vials_affected=n if err > self.tol else 0,
                    advice=(f"Expect the first {n} vial(s) of {nxt.name} to be off-colour; "
                            f"flush ~{self.hold_up:.0f} mL to waste before the changeover"
                            if err > self.tol else "No flush needed"))

    # ---------------------------------------------------------------- verdict
    def judge(self, observed, r: Recipe, history: list, first_obs=None,
              pos_in_batch: int | None = None) -> dict:
        """Compare a measured vial with its recipe and explain any mismatch."""
        obs = np.asarray(observed, float)
        clean = self.clean(r)
        dirty = self.with_history(history, r)
        ec = float(np.linalg.norm(obs - clean))
        ed = float(np.linalg.norm(obs - dirty))
        prev = next((h for h in reversed(history) if h.name != r.name), None)
        changeover = bool(history) and history[-1].name != r.name

        reasons = []
        if ec <= self.tol:
            status = "OK"
            reasons.append("colour matches the recipe")
        elif ed <= ec * (1 - self.explain_gain) and prev is not None:
            status = "CARRYOVER"
            reasons.append(f"off-colour; residue from {prev.name} in the shared mixing path "
                           f"explains it (error {ec:.2f} -> {ed:.2f} once carryover is modelled)")
        elif changeover and prev is not None:
            status = "CARRYOVER?"
            reasons.append(f"off-colour on the first vial after a changeover from {prev.name}; "
                           f"likely residue, but the model does not fully reproduce it")
        else:
            status = "MISMATCH"
            reasons.append("off-colour and NOT explained by the previous recipe - check tank "
                           "contents, dosing volumes and lighting")

        # Direct evidence: the first liquid to land should be this recipe's first tank.
        if first_obs is not None and prev is not None and status != "OK":
            fo = np.asarray(first_obs, float)
            d_prev = float(np.linalg.norm(fo - self.clean(prev)))
            d_self = float(np.linalg.norm(fo - clean))
            if d_prev < d_self:
                reasons.append(f"first liquid into the vial looked like {prev.name} "
                               f"({hexcolor(fo)}), not this recipe")
        if pos_in_batch == 1 and changeover and status == "OK":
            reasons.append("first vial after a changeover came out clean")

        return dict(status=status, ok=status == "OK", error_clean=round(ec, 3),
                    error_carryover=round(ed, 3), observed_hex=hexcolor(obs),
                    expected_hex=hexcolor(clean), predicted_hex=hexcolor(dirty),
                    previous=prev.name if prev else None, changeover=changeover,
                    reasons=reasons)
