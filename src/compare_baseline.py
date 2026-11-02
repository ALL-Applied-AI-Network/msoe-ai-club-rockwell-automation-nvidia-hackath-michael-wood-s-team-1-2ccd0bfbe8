"""Compare the recipe-aware check with a color-library check like the line's current inspection.

The baseline passes any vial whose color is close to the clean color of some product the line makes
(the last vial of each batch), which is what a "known color" check does. It never looks at the recipe.
Both checks are scored on the same labeled vials, with leave-one-recipe-out for ours.

    python src/compare_baseline.py                 # uses out/events_video5.csv if present
    python src/compare_baseline.py --refresh       # re-run vial detection on video 5 first
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_model import detect, load_sequence, evaluate, truth, metrics

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--refresh", action="store_true")
ap.add_argument("--tolerance", type=float, default=0.30, help="alarm threshold for both checks")
ap.add_argument("--truth-tol", type=float, default=0.25)
a = ap.parse_args()

seq, _ = load_sequence(detect(a.refresh))
by_rid = {s["rid"]: s for s in seq if s["last"]}

rows, ours, cause = evaluate(seq, a.tolerance, 0.30, a.truth_tol)

library = [s["obs"] for s in seq if s["last"] and s["obs"] is not None]
t, p = [], []
for s in seq:
    d, _ = truth(s, by_rid, a.truth_tol)
    if d is None:
        continue
    nearest = min(np.linalg.norm(s["obs"] - c) for c in library if c is not s["obs"])
    t.append(d); p.append(nearest > a.tolerance)
base = metrics(t, p)

print(f"threshold {a.tolerance}")
print(f"{'':32}{'color library':>16}{'recipe-aware':>16}")
print(f"{'off-color vials caught':32}{base['tp']:>10} of {base['tp'] + base['fn']:<3}{ours['tp']:>10} of {ours['tp'] + ours['fn']:<3}")
print(f"{'false alarms on clean vials':32}{base['fp']:>10} of {base['fp'] + base['tn']:<3}{ours['fp']:>10} of {ours['fp'] + ours['tn']:<3}")
print(f"{'correct cause given':32}{'none':>16}{cause['correct']:>10} of {cause['n']:<3}")
