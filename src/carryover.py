"""Expected-colour and carryover model for the fluid mixing station (Objective 2).

Predicts what colour a vial SHOULD be from its recipe, predicts how much of the
previous recipe will contaminate it, and fits both to the real measurements.

Three things it does differently from assuming a model and moving on:

1. **The clear tank is pinned to zero.** Recipe fractions always sum to 1, so a
   per-vial baseline and the three dye coefficients are collinear - you cannot tell
   "every vial absorbs B" from "every tank absorbs B more". Defining T-100 (the clear
   diluent) as zero dye breaks the tie and makes the remaining baseline identifiable
   as what it physically is: water and glass against an air-filled reference.

2. **Dispense order is simulated, not averaged.** The shared path is a delay line of
   volume V_d. Fluids enter in the fixed T-200 -> T-400 -> T-100 order, so whatever is
   sitting in the line when a batch ends is the *last* thing dispensed. That is why
   order matters: a recipe ending on clear leaves a clean line, one ending on dye
   does not.

3. **Two hydraulics are fitted and compared.** Plug flow (the new fluid pushes the old
   out as a slug) versus well-mixed (the line is a stirred tank). These predict
   different things and the data picks the winner.

Given V_d the model is linear in the dye coefficients, so V_d is found by profiling:
scan V_d, solve least squares at each, keep the best.

    python src/carryover.py                 # fit, compare models, report
    python src/carryover.py --explain 7     # explain one vial
    python src/carryover.py --risk 6 7      # risk of running recipe 7 after recipe 6
"""
from __future__ import annotations

import argparse, csv, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VF = os.path.join(ROOT, "data", "vial_fill")
OUT = os.path.join(ROOT, "out")
TANKS = ("T-100", "T-200", "T-400")
DISPENSE_ORDER = ("T-200", "T-400", "T-100")     # fixed by the station
SLICE_ML = 0.25                                  # delay-line discretisation


# ----------------------------------------------------------------- data
# Detected fill k is matched to labelled vial k. That holds only while no fill is
# missed, and segment_fills finds 27 of the 30 vials in video 5. Checking each fill's
# measured hue against its recipe's expected character shows the mapping is sound
# through vial 18 and desynchronises after it (Recipe 11 should be blue but reads
# green, Recipe 12 should be orange but reads blue). Everything past 18 is therefore
# excluded from the fit rather than silently corrupting it.
TRUSTED_THROUGH_VIAL = 18


def load():
    fills = {int(f["fill"]): f for f in
             csv.DictReader(open(os.path.join(OUT, "fills_video5.csv")))}
    recipes = {int(r["vial"]): r for r in
               csv.DictReader(open(os.path.join(VF, "labels", "video5_vial_recipes.csv")))}
    seq = []
    for vial in sorted(recipes):
        r = recipes[vial]
        vols = {t: float(r[k] or 0) for t, k in
                (("T-100", "t100_ml"), ("T-200", "t200_ml"), ("T-400", "t400_ml"))}
        f = fills.get(vial)                      # matched by order; see calibrate_tanks.py
        obs = np.array([float(f["a_r"]), float(f["a_g"]), float(f["a_b"])]) if f else None
        if vial > TRUSTED_THROUGH_VIAL:
            obs = None                       # mapping not verifiable past here
        seq.append(dict(vial=vial, recipe=int(r["recipe_id"]), vols=vols,
                        total=sum(vols.values()), pos=(vial - 1) % 3, obs=obs))
    return seq


# ----------------------------------------------- hydraulics of the shared path
def simulate(seq, hold_up_ml: float, well_mixed: bool):
    """Push each recipe through the shared path; return the mix that lands in each vial.

    Composition is carried as fractions of (T-100, T-200, T-400). The line starts
    pre-loaded with the first recipe so the opening vial is not spuriously flagged.
    """
    n_slices = max(1, int(round(hold_up_ml / SLICE_ML)))
    line = np.zeros((n_slices, 3))
    first = np.array([seq[0]["vols"][t] for t in TANKS], float)
    line[:] = first / max(first.sum(), 1e-9)      # warm start

    out = []
    for item in seq:
        collected = np.zeros(3)
        for tank in DISPENSE_ORDER:
            volume = item["vols"][tank]
            if volume <= 0:
                continue
            inlet = np.zeros(3); inlet[TANKS.index(tank)] = 1.0
            for _ in range(int(round(volume / SLICE_ML))):
                if well_mixed:
                    collected += line[0] * SLICE_ML
                    alpha = min(1.0, SLICE_ML / hold_up_ml)   # clamp: a tiny line flushes fully
                    line[:] = line[0] + (inlet - line[0]) * alpha
                else:
                    collected += line[-1] * SLICE_ML       # slug exits the far end
                    line[1:] = line[:-1]
                    line[0] = inlet
        total = collected.sum()
        out.append(collected / total if total > 0 else collected)
    return np.array(out)


def design(mix: np.ndarray) -> np.ndarray:
    """Rows [1, f_T200, f_T400]: a constant water/glass baseline plus the two dyes.

    T-100 is the clear diluent and is defined as zero dye, which is what makes the
    baseline identifiable instead of collinear with the tank coefficients.
    """
    return np.column_stack([np.ones(len(mix)), mix[:, 1], mix[:, 2]])


def fit(seq, well_mixed: bool, grid=np.arange(0.5, 80.01, 0.5)):
    """Two stages, so contamination cannot bias the colour model.

    Stage 1 fits the dye coefficients on 3rd-of-batch vials only - the ones the data
    shows are actually clean. Fitting them on every vial lets the contaminated firsts
    drag every prediction toward the mean and leaves clean vials under-saturated.

    Stage 2 holds those coefficients fixed and scans the hold-up volume, which is the
    only remaining free parameter.
    """
    clean_idx = [i for i, s_ in enumerate(seq) if s_["obs"] is not None and s_["pos"] == 2]
    Yc = np.array([seq[i]["obs"] for i in clean_idx])
    Xc = design(np.array([[seq[i]["vols"][t] / seq[i]["total"] for t in TANKS]
                          for i in clean_idx]))
    coef, *_ = np.linalg.lstsq(Xc, Yc, rcond=None)
    dye_rms = float(np.sqrt(((Yc - Xc @ coef) ** 2).mean()))

    keep = [i for i, s_ in enumerate(seq) if s_["obs"] is not None]
    Y = np.array([seq[i]["obs"] for i in keep])
    best = None
    for v_d in grid:
        mix = simulate(seq, float(v_d), well_mixed)[keep]
        rms = float(np.sqrt(((Y - design(mix) @ coef) ** 2).mean()))
        if best is None or rms < best["rms"]:
            best = dict(hold_up_ml=float(v_d), coef=coef, rms=rms,
                        well_mixed=well_mixed, dye_rms=dye_rms,
                        at_grid_edge=bool(v_d >= grid[-1] - 1e-9))
    return best


def predict(seq, model, clean: bool):
    """Expected absorbance per vial. clean=True ignores carryover (what it SHOULD be)."""
    if clean:
        mix = np.array([[s["vols"][t] / s["total"] for t in TANKS] for s in seq])
    else:
        mix = simulate(seq, model["hold_up_ml"], model["well_mixed"])
    return design(mix) @ model["coef"], mix


# --------------------------------------------------------------- reporting
def to_rgb(absorbance):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vialcolor import linear_to_srgb
    return tuple(int(v) for v in linear_to_srgb(np.power(10.0, -np.maximum(absorbance, 0))))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--explain", type=int, metavar="VIAL")
    ap.add_argument("--risk", nargs=2, type=int, metavar=("PREV", "NEXT"))
    a = ap.parse_args()

    seq = load()
    print(f"\nUsing vials 1-{TRUSTED_THROUGH_VIAL}: past that the detected-fill to "
          f"labelled-vial mapping desynchronises (3 fills missed).")
    plug, mixed = fit(seq, well_mixed=False), fit(seq, well_mixed=True)
    model = min((plug, mixed), key=lambda m: m["rms"])
    name = "well-mixed" if model["well_mixed"] else "plug-flow"

    if a.risk:
        prev_id, next_id = a.risk
        byr = {}
        for s in seq:
            byr.setdefault(s["recipe"], s)
        if prev_id not in byr or next_id not in byr:
            ap.error(f"recipes available: {sorted(byr)}")
        pair = [byr[prev_id], byr[prev_id], byr[prev_id], byr[next_id]]
        mix = simulate(pair, model["hold_up_ml"], model["well_mixed"])
        dirty = design(mix[-1:]) @ model["coef"]
        clean_frac = np.array([[byr[next_id]["vols"][t] / byr[next_id]["total"] for t in TANKS]])
        clean = design(clean_frac) @ model["coef"]
        carry = 1.0 - mix[-1] @ clean_frac[0] / max(np.linalg.norm(clean_frac[0]) ** 2, 1e-9)
        dE = float(np.linalg.norm(dirty - clean))
        print(f"\nRunning Recipe {next_id} after Recipe {prev_id}")
        print(f"  expected if clean : RGB {to_rgb(clean[0])}")
        print(f"  predicted 1st vial: RGB {to_rgb(dirty[0])}")
        print(f"  absorbance error  : {dE:.3f}   contamination {abs(carry)*100:.0f}%")
        if dE > 0.15:
            print(f"  verdict           : FLUSH FIRST - run ~{model['hold_up_ml']:.0f} mL "
                  f"to waste, or expect to scrap the first vial\n")
        else:
            print("  verdict           : acceptable, no flush needed\n")
        return

    if a.explain:
        pred_dirty, mix = predict(seq, model, clean=False)
        pred_clean, _ = predict(seq, model, clean=True)
        i = next((j for j, s_ in enumerate(seq) if s_["vial"] == a.explain), None)
        if i is None:
            ap.error(f"vial must be 1-{len(seq)}")
        s_ = seq[i]
        print(f"\nVial {s_['vial']}  (Recipe {s_['recipe']}, "
              f"{['1st','2nd','3rd'][s_['pos']]} of its batch)")
        order = " -> ".join(t for t in DISPENSE_ORDER if s_["vols"][t] > 0)
        print(f"  recipe        : " + ", ".join(f"{t} {s_['vols'][t]:.0f} mL"
                                                for t in TANKS if s_["vols"][t] > 0)
              + f"   (dispensed {order})")
        print(f"  should look   : RGB {to_rgb(pred_clean[i])}")
        if s_["obs"] is None:
            print("  measured      : no trusted measurement for this vial"); return
        ec = float(np.linalg.norm(s_["obs"] - pred_clean[i]))
        ed = float(np.linalg.norm(s_["obs"] - pred_dirty[i]))
        prev = seq[i-1] if i else None
        print(f"  measured      : RGB {to_rgb(s_['obs'])}   error vs clean {ec:.3f}")
        print(f"  with carryover: RGB {to_rgb(pred_dirty[i])}   error {ed:.3f}")
        carry = 1.0 - mix[i] @ np.array([s_["vols"][t]/s_["total"] for t in TANKS])
        if ec < 0.25:
            print("  VERDICT       : within tolerance, no action")
        elif ed < ec * 0.7:
            print(f"  VERDICT       : carryover from Recipe {prev['recipe'] if prev else '?'} "
                  f"explains it (error falls {100*(ec-ed)/ec:.0f}% when the previous "
                  f"recipe's residue is included)")
            print(f"  ACTION        : flush ~{model['hold_up_ml']:.0f} mL before the next changeover")
        else:
            print("  VERDICT       : off-colour, and carryover does NOT explain it - "
                  "check tank contents and dosing logs")
        return

    pred_dirty, mix = predict(seq, model, clean=False)
    pred_clean, _ = predict(seq, model, clean=True)

    print(f"\nHydraulic model comparison (fitted on {sum(s['obs'] is not None for s in seq)} vials)")
    print(f"  dye coefficients fitted on {sum(1 for s_ in seq if s_['pos']==2 and s_['obs'] is not None)} "
          f"clean (3rd-of-batch) vials, RMS {plug['dye_rms']:.4f}")
    print(f"  plug-flow   V_d = {plug['hold_up_ml']:5.1f} mL   RMS {plug['rms']:.4f}"
          + ("  [at grid edge]" if plug["at_grid_edge"] else ""))
    print(f"  well-mixed  V_d = {mixed['hold_up_ml']:5.1f} mL   RMS {mixed['rms']:.4f}"
          + ("  [at grid edge]" if mixed["at_grid_edge"] else ""))
    print(f"  -> {name} wins by {abs(plug['rms']-mixed['rms'])/max(plug['rms'],mixed['rms'])*100:.1f}%\n")

    base, e200, e400 = model["coef"]
    print("Fitted dye coefficients (absorbance R/G/B at full strength)")
    print(f"  water + glass baseline  {np.round(base,3)}")
    print(f"  T-200                   {np.round(e200,3)}   RGB {to_rgb(base+e200)}")
    print(f"  T-400                   {np.round(e400,3)}   RGB {to_rgb(base+e400)}")
    print(f"  T-100 (clear)           pinned to zero\n")

    print(f"{'vial':>4} {'rcp':>3} {'pos':>4} {'observed':>14} {'expected':>14} "
          f"{'w/carry':>14} {'err_clean':>9} {'err_carry':>9}  verdict")
    print("-" * 104)
    rows = []
    for i, s in enumerate(seq):
        if s["obs"] is None:
            continue
        ec = float(np.linalg.norm(s["obs"] - pred_clean[i]))
        ed = float(np.linalg.norm(s["obs"] - pred_dirty[i]))
        verdict = "OK" if ec < 0.25 else ("carryover explains it" if ed < ec * 0.7 else "unexplained")
        print(f"{s['vial']:>4} {s['recipe']:>3} {['1st','2nd','3rd'][s['pos']]:>4} "
              f"{str(to_rgb(s['obs'])):>14} {str(to_rgb(pred_clean[i])):>14} "
              f"{str(to_rgb(pred_dirty[i])):>14} {ec:>9.3f} {ed:>9.3f}  {verdict}")
        rows.append((s, ec, ed, verdict))

    firsts = [r for r in rows if r[0]["pos"] == 0]
    later = [r for r in rows if r[0]["pos"] != 0]
    expl = sum(1 for r in firsts if r[3] == "carryover explains it")
    fa = sum(1 for r in later if r[3] != "OK")
    print("-" * 104)
    print(f"first-of-batch vials explained by carryover : {expl}/{len(firsts)}")
    print(f"later vials flagged (false alarms)          : {fa}/{len(later)}")
    print(f"mean error, no carryover model              : {np.mean([r[1] for r in rows]):.3f}")
    print(f"mean error, with carryover model            : {np.mean([r[2] for r in rows]):.3f}")

    with open(os.path.join(OUT, "carryover_model.json"), "w") as fh:
        json.dump({"hydraulics": name, "hold_up_ml": model["hold_up_ml"], "rms": model["rms"],
                   "baseline": base.tolist(), "T-200": e200.tolist(), "T-400": e400.tolist()},
                  fh, indent=2)
    print("\nwrote out/carryover_model.json")


if __name__ == "__main__":
    main()
