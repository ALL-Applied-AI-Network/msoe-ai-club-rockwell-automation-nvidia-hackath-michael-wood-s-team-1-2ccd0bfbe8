"""Feed our real measurements into the PartTwo RecipeAnalysis model.

Our pipeline supplies the two things that model needs and does not have:
measured tank colours (src/calibrate_tanks.py) and observed vial colours
(src/segment_fills.py). This runs it over every recipe transition in video 5.

    python src/run_integration.py --model <path to RecipeAnalysis.py> [--hold-up 20]
"""
import argparse, csv, importlib.util, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
VF = os.path.join(ROOT, "data", "vial_fill")

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="path to RecipeAnalysis.py from the PartTwo branch")
ap.add_argument("--hold-up", type=float, default=20.0, help="shared-path dead volume, mL")
ap.add_argument("--tolerance", type=float, default=5.0)
a = ap.parse_args()

spec = importlib.util.spec_from_file_location("RecipeAnalysis", a.model)
RA = importlib.util.module_from_spec(spec)
sys.modules["RecipeAnalysis"] = RA        # dataclasses resolves annotations via sys.modules
spec.loader.exec_module(RA)

cal = json.load(open(os.path.join(OUT, "tank_calibration.json")))
fills = {int(f["fill"]): f for f in csv.DictReader(open(os.path.join(OUT, "fills_video5.csv")))}
recipes = {int(r["vial"]): r for r in
           csv.DictReader(open(os.path.join(VF, "labels", "video5_vial_recipes.csv")))}

def volumes(r):
    v = {t: float(r[k] or 0) for t, k in
         (("T-100", "t100_ml"), ("T-200", "t200_ml"), ("T-400", "t400_ml"))}
    return {t: x for t, x in v.items() if x > 0}

batches = {}
for vial, r in recipes.items():
    batches.setdefault(int(r["recipe_id"]), []).append(vial)
order = sorted(batches)

print(f"tank calibration: " + "  ".join(f"{t}={cal['tank_rgb'][t]}" for t in cal["tank_rgb"]))
print(f"hold-up volume: {a.hold_up} mL   tolerance: dE {a.tolerance}\n")
print(f"{'rcp':>4} {'vial':>4} {'pos':>4} {'observed':>10} {'expected':>10} {'dE':>6} "
      f"{'frac':>5}  status")
print("-" * 86)

hits = miss = fp = tn = 0
for i, rid in enumerate(order):
    if i == 0:
        continue                                   # no previous recipe to carry over from
    prev_r = recipes[batches[order[i - 1]][0]]
    cur_r = recipes[batches[rid][0]]
    obs = []
    for vial in sorted(batches[rid]):
        f = fills.get(vial)
        obs.append([float(f["r"]), float(f["g"]), float(f["b"])] if f else None)
    payload = {
        "tank_rgb": cal["tank_rgb"],
        "previous": {"name": f"Recipe {order[i-1]}", "volumes_ml": volumes(prev_r)},
        "current": {"name": f"Recipe {rid}", "volumes_ml": volumes(cur_r)},
        "hold_up_ml": a.hold_up, "tolerance_delta_e": a.tolerance,
        "observations_rgb": [o for o in obs if o],
    }
    try:
        res = RA.analyze(payload)
    except Exception as exc:
        print(f"{rid:>4}  model error: {exc}"); continue

    vials = sorted(batches[rid])
    for n, v in enumerate(res["vials"]):
        vial = vials[n]
        pos = ["1st", "2nd", "3rd"][n]
        rgbo = tuple(int(x) for x in v["observed_rgb"])
        rgbe = tuple(int(x) for x in v["expected_rgb"])
        print(f"{rid:>4} {vial:>4} {pos:>4} {str(rgbo):>10} {str(rgbe):>10} "
              f"{v['delta_e_expected']:>6.1f} {v['fitted_residual_fraction']:>5.2f}  {v['status']}")
        first = (n == 0)
        carry = v["status"] == "carryover_consistent"
        if first and carry: hits += 1
        elif first and not carry: miss += 1
        elif not first and v["color_defect"]: fp += 1
        else: tn += 1
    t = res["transition"]
    print(f"{'':>4} {'':>4} {'':>4}  transition risk {t['score_0_100']:>5.1f}  "
          f"warning={t['warning']}\n")

print("-" * 86)
print(f"first-of-batch explained as carryover: {hits}/{hits+miss}")
print(f"later vials flagged as defective (false alarms): {fp}/{fp+tn}")
