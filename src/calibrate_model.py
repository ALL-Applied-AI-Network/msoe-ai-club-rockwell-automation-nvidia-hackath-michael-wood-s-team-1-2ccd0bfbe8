"""Fit the frozen colour model and score it with leave-one-recipe-out evaluation.

    python src/calibrate_model.py                 # fit + evaluate, write models/color_model.json
    python src/calibrate_model.py --refresh       # re-run vial detection on the video first

Steps
1. Detect every vial in vial_fill video 5 with src/fillevents.py (no labels used).
2. Match each detection to a labelled vial by TIME OVERLAP with its fill window.
   At 30 fps the label windows and the detections agree to about a second, so no
   ordering assumption is needed and missed vials stay missing instead of shifting
   every later match.
3. Fit the tank dye coefficients on the last vial of each batch (the event guide's
   reliable clean colour), then the shared-path hold-up volume on all vials.
4. Leave-one-recipe-out: for each recipe, refit with that recipe's vials removed and
   judge them. Truth is independent of the model: a vial is DEFECTIVE if its colour
   differs from the last vial of its own batch by more than --truth-tol, and its cause
   is CARRYOVER if it is the 1st/2nd vial after a recipe change.

False alarms and missed defects are reported separately, and the cause explanation
is scored separately from detection, as the event guide asks.
"""
from __future__ import annotations

import argparse, csv, json, os, subprocess, sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from carryover import TANKS, simulate, design
from colormodel import ColorModel, Recipe, hexcolor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VF = os.path.join(ROOT, "data", "vial_fill")
OUT = os.path.join(ROOT, "out")
MODELS = os.path.join(ROOT, "models")
EVENTS_CSV = os.path.join(OUT, "events_video5.csv")
FIELDS = ["index", "start_s", "end_s", "start_ml", "final_ml", "a_r", "a_g", "a_b",
          "f_r", "f_g", "f_b", "hue_deg", "first_hue_deg", "od"]


def detect(refresh: bool):
    if refresh or not os.path.exists(EVENTS_CSV):
        from fillevents import events_from_video
        print("detecting vials in video 5 (about 40 s)...")
        evs = events_from_video(os.path.join(VF, "videos", "vial_fill_video_5.mp4"))
        os.makedirs(OUT, exist_ok=True)
        with open(EVENTS_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
            for e in evs:
                w.writerow(dict(index=e.index, start_s=e.start_s, end_s=e.end_s,
                                start_ml=e.start_ml, final_ml=e.final_ml,
                                a_r=e.absorbance[0], a_g=e.absorbance[1], a_b=e.absorbance[2],
                                f_r=e.first_absorbance[0], f_g=e.first_absorbance[1],
                                f_b=e.first_absorbance[2], hue_deg=e.hue_deg,
                                first_hue_deg=e.first_hue_deg, od=e.od))
    return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(EVENTS_CSV))]


def load_sequence(events):
    """Labelled vials in order, each with its matched detection (or None)."""
    labels = list(csv.DictReader(open(os.path.join(VF, "labels", "video5_vial_recipes.csv"))))
    seq, used = [], set()
    for r in labels:
        s, e = float(r["start_s"]), float(r["end_s"])
        best, best_ov = None, 0.0
        for ev in events:
            ov = min(e, ev["end_s"]) - max(s, ev["start_s"])
            if ov > best_ov and ev["index"] not in used:
                best, best_ov = ev, ov
        if best is not None and best_ov < 0.3 * (e - s):
            best = None                                   # overlap too small to trust
        if best is not None:
            used.add(best["index"])
        rec = Recipe(name=f"Recipe {r['recipe_id']}", t100=float(r["t100_ml"] or 0),
                     t200=float(r["t200_ml"] or 0), t400=float(r["t400_ml"] or 0))
        seq.append(dict(vial=int(r["vial"]), recipe=rec, rid=int(r["recipe_id"]),
                        vols=rec.vols, total=rec.total,
                        obs=None if best is None else np.array([best["a_r"], best["a_g"], best["a_b"]]),
                        first=None if best is None else np.array([best["f_r"], best["f_g"], best["f_b"]]),
                        det=best))
    # Position in batch and changeover flags.
    for i, s in enumerate(seq):
        s["pos"] = 1 if i == 0 or seq[i - 1]["rid"] != s["rid"] else seq[i - 1]["pos"] + 1
    last = {}
    for s in seq:
        last[s["rid"]] = s
    for s in seq:
        s["last"] = last[s["rid"]] is s
    return seq, [e for e in events if e["index"] not in used]


def fit(seq, train, grid=np.arange(1.0, 80.01, 1.0)):
    """Dye coefficients on clean (last-of-batch) training vials, then hold-up volume."""
    clean = [s for s in seq if s["rid"] in train and s["last"] and s["obs"] is not None]
    X = design(np.array([[s["vols"][t] / s["total"] for t in TANKS] for s in clean]))
    Y = np.array([s["obs"] for s in clean])
    if len(clean) < 3 or np.linalg.matrix_rank(X) < 3:
        raise ValueError("not enough independent clean recipes to fit the dye coefficients")
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    idx = [i for i, s in enumerate(seq) if s["rid"] in train and s["obs"] is not None]
    Yall = np.array([seq[i]["obs"] for i in idx])
    best = None
    for wm in (True, False):
        for v in grid:
            mix = simulate(seq, float(v), wm)[idx]
            rms = float(np.sqrt(((Yall - design(mix) @ coef) ** 2).mean()))
            if best is None or rms < best[0]:
                best = (rms, float(v), wm)
    return dict(baseline=coef[0].tolist(), **{"T-200": coef[1].tolist(), "T-400": coef[2].tolist()},
                hold_up_ml=best[1], hydraulics="well-mixed" if best[2] else "plug-flow",
                fit_rms=round(best[0], 4), n_clean=len(clean))


def truth(s, by_rid, truth_tol):
    ref = by_rid[s["rid"]]["obs"]
    if s["obs"] is None or ref is None:
        return None, None
    defect = (not s["last"]) and float(np.linalg.norm(s["obs"] - ref)) > truth_tol
    cause = "carryover" if defect and s["pos"] <= 2 and s["vial"] > 3 else ("other" if defect else "clean")
    return defect, cause


def metrics(t, p):
    tp = sum(a and b for a, b in zip(t, p)); fp = sum((not a) and b for a, b in zip(t, p))
    fn = sum(a and not b for a, b in zip(t, p)); tn = sum((not a) and not b for a, b in zip(t, p))
    r = lambda a, b: round(a / b, 3) if b else None
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=r(tp, tp + fp), recall=r(tp, tp + fn),
                false_alarm_rate=r(fp, fp + tn), accuracy=r(tp + tn, tp + fp + fn + tn))


def evaluate(seq, tol, gain, truth_tol):
    rids = sorted({s["rid"] for s in seq})
    by_rid = {s["rid"]: s for s in seq if s["last"]}
    rows = []
    for held in rids:
        train = [r for r in rids if r != held]
        params = fit(seq, train)
        params.update(tolerance=tol, explain_gain=gain)
        m = ColorModel(params)
        for i, s in enumerate(seq):
            if s["rid"] != held or s["obs"] is None:
                continue
            hist = [x["recipe"] for x in seq[:i]]
            v = m.judge(s["obs"], s["recipe"], hist, first_obs=s["first"], pos_in_batch=s["pos"])
            d, c = truth(s, by_rid, truth_tol)
            rows.append(dict(vial=s["vial"], recipe=held, pos=s["pos"], status=v["status"],
                             flagged=not v["ok"], carry_claim=v["status"].startswith("CARRYOVER"),
                             err_clean=v["error_clean"], err_carry=v["error_carryover"],
                             observed=v["observed_hex"], expected=v["expected_hex"],
                             predicted=v["predicted_hex"], truth_defect=d, truth_cause=c,
                             reasons=" | ".join(v["reasons"])))
    scored = [r for r in rows if r["truth_defect"] is not None]
    det = metrics([r["truth_defect"] for r in scored], [r["flagged"] for r in scored])
    # Cause is scored only on vials the system flagged AND that truly were defective.
    both = [r for r in scored if r["flagged"] and r["truth_defect"]]
    cause = dict(n=len(both), correct=sum((r["truth_cause"] == "carryover") == r["carry_claim"] for r in both))
    cause["accuracy"] = round(cause["correct"] / cause["n"], 3) if cause["n"] else None
    return rows, det, cause


def git_rev():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-run vial detection on the video")
    ap.add_argument("--tolerance", type=float, default=0.25, help="clean-vs-observed alarm threshold")
    ap.add_argument("--explain-gain", type=float, default=0.30)
    ap.add_argument("--truth-tol", type=float, default=0.25, help="defect = differs from batch's last vial by more")
    ap.add_argument("--sweep", action="store_true", help="also sweep the alarm threshold")
    a = ap.parse_args()

    seq, extra = load_sequence(detect(a.refresh))
    matched = sum(s["obs"] is not None for s in seq)
    print(f"labelled vials: {len(seq)}   matched to a detection: {matched}   "
          f"unmatched detections: {len(extra)}")

    rows, det, cause = evaluate(seq, a.tolerance, a.explain_gain, a.truth_tol)
    print(f"\nleave-one-recipe-out, tolerance {a.tolerance}")
    print(f"{'vial':>4} {'rcp':>3} {'pos':>3}  {'obs':>7} {'exp':>7} {'pred':>7} {'e_cln':>5} {'e_car':>5}  "
          f"{'truth':>9}  verdict")
    for r in rows:
        t = "-" if r["truth_defect"] is None else (r["truth_cause"])
        print(f"{r['vial']:>4} {r['recipe']:>3} {r['pos']:>3}  {r['observed']} {r['expected']} {r['predicted']} "
              f"{r['err_clean']:>5.2f} {r['err_carry']:>5.2f}  {t:>9}  {r['status']}")
    print(f"\ndetection : {det}")
    print(f"cause     : {cause}")

    if a.sweep:
        print("\nthreshold sweep (leave-one-recipe-out)")
        for tol in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
            _, d, c = evaluate(seq, tol, a.explain_gain, a.truth_tol)
            print(f"  tol {tol:.2f}  recall {d['recall']}  false-alarm {d['false_alarm_rate']}  "
                  f"precision {d['precision']}  cause-acc {c['accuracy']}")

    # Frozen deployment model: fitted on every recipe.
    params = fit(seq, sorted({s["rid"] for s in seq}))
    params.update(tolerance=a.tolerance, explain_gain=a.explain_gain,
                  fitted_on="vial_fill video 5, last vial of each batch (dye) + all vials (hold-up)",
                  created=datetime.now().isoformat(timespec="seconds"), code_rev=git_rev())
    os.makedirs(MODELS, exist_ok=True)
    with open(os.path.join(MODELS, "color_model.json"), "w") as f:
        json.dump(params, f, indent=2)
    with open(os.path.join(OUT, "evaluation_video5.json"), "w") as f:
        json.dump(dict(detection=det, cause=cause, rows=rows, tolerance=a.tolerance,
                       truth_tol=a.truth_tol, matched=matched, n_labelled=len(seq)), f, indent=2,
                  default=lambda o: o if not isinstance(o, np.generic) else o.item())
    print(f"\nfrozen model: hold-up {params['hold_up_ml']} mL ({params['hydraulics']}), "
          f"fit RMS {params['fit_rms']} -> models/color_model.json")


if __name__ == "__main__":
    main()
