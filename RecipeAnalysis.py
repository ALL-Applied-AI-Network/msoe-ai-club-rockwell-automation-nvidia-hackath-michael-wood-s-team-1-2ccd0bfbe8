"""Fluid-mixing inspection prototype (Python 3.10+, standard library only).

Run: python RecipeAnalysis.py
     python RecipeAnalysis.py --input inspection.json
See RecipeAnalysisGuide.md for assumptions, formulas, and input format.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from typing import Sequence

ORDER = ("T-200", "T-400", "T-100")
RGB = tuple[float, float, float]
Vector = tuple[float, float, float]
# Illustrative sRGB samples under a white background, NOT measured tank colors.
DEMO_COLORS = {"T-100": (255, 255, 255), "T-200": (240, 125, 30),
               "T-400": (35, 100, 225)}


def number(value: float, name: str, minimum: float = 0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return value


def rgb(value: Sequence[float]) -> RGB:
    if len(value) != 3:
        raise ValueError("RGB must contain three channels in [0, 255]")
    result = tuple(number(x, "RGB channel") for x in value)
    if any(x > 255 for x in result):
        raise ValueError("RGB channels must be <= 255")
    return result


def linearize(x: float) -> float:
    x /= 255
    return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4


def encode(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return 255 * (12.92 * x if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055)


def optical_density(color: RGB) -> Vector:
    return tuple(-math.log10(max(1e-6, linearize(c))) for c in rgb(color))


def from_density(density: Vector) -> RGB:
    return tuple(encode(10 ** (-x)) for x in density)


def blend(a: Vector, b: Vector, fraction_b: float) -> Vector:
    return tuple((1 - fraction_b) * x + fraction_b * y for x, y in zip(a, b))


def lab(color: RGB) -> Vector:
    r, g, b = (linearize(c) for c in rgb(color))
    # sRGB -> XYZ (D65), then CIELAB using the matching D65 white.
    xyz = ((0.4124564*r + 0.3575761*g + 0.1804375*b) / 0.95047,
           0.2126729*r + 0.7151522*g + 0.0721750*b,
           (0.0193339*r + 0.1191920*g + 0.9503041*b) / 1.08883)
    delta = 6 / 29
    f = tuple(t ** (1/3) if t > delta**3 else t/(3*delta**2) + 4/29 for t in xyz)
    return (116*f[1]-16, 500*(f[0]-f[1]), 200*(f[1]-f[2]))


def delta_e(a: RGB, b: RGB) -> float:
    return math.dist(lab(a), lab(b))


@dataclass(frozen=True)
class Recipe:
    name: str
    volumes_ml: dict[str, float]

    def __post_init__(self):
        volumes = dict(self.volumes_ml)
        if "T-300" in volumes:
            raise ValueError("T-300 is out of service; remove it even if its volume is zero")
        if set(volumes) - set(ORDER):
            raise ValueError("Unknown tank in recipe")
        volumes = {tank: number(v, f"{tank} volume") for tank, v in volumes.items()}
        if not math.isfinite(sum(volumes.values())) or sum(volumes.values()) <= 0:
            raise ValueError("Recipe must dispense a positive volume")
        object.__setattr__(self, "volumes_ml", volumes)

    @property
    def total_ml(self) -> float:
        return sum(self.volumes_ml.values())


class MixingModel:
    def __init__(self, tank_rgb: dict[str, Sequence[float]]):
        if set(tank_rgb) != set(ORDER):
            raise ValueError("Calibration must include exactly T-100, T-200, T-400")
        self.densities = {tank: optical_density(rgb(c)) for tank, c in tank_rgb.items()}

    def expected_density(self, recipe: Recipe) -> Vector:
        return tuple(sum(recipe.volumes_ml.get(tank, 0) * self.densities[tank][i]
                         for tank in ORDER) / recipe.total_ml for i in range(3))

    def expected_rgb(self, recipe: Recipe) -> RGB:
        return from_density(self.expected_density(recipe))

    def dispense(self, recipe: Recipe, initial: Vector, hold_up_ml: float) -> tuple[RGB, Vector]:
        """Well-mixed shared path; integrate outlet density over the full vial.

        Inlet tank concentrations change in ORDER. Equal flow displacement and
        fixed path volume are assumed. Return collected color and final path state.
        """
        h = number(hold_up_ml, "hold_up_ml")
        state = initial
        collected = [0.0, 0.0, 0.0]
        for tank in ORDER:
            volume = recipe.volumes_ml.get(tank, 0)
            if not volume:
                continue
            inlet = self.densities[tank]
            end = blend(inlet, state, math.exp(-volume / h)) if h else inlet
            for i in range(3):
                collected[i] += inlet[i]*volume - h*(end[i]-state[i])
            state = end
        return from_density(tuple(max(0.0, x / recipe.total_ml) for x in collected)), state

    def settled_tail(self, recipe: Recipe, hold_up_ml: float) -> Vector:
        """Solve the periodic path state analytically for repeated recipe vials."""
        h = number(hold_up_ml, "hold_up_ml")
        _, tail_from_zero = self.dispense(recipe, (0, 0, 0), h)
        denominator = -math.expm1(-recipe.total_ml/h) if h else 1.0
        return tuple(x / denominator for x in tail_from_zero)


def explain(model: MixingModel, recipe: Recipe, observed: RGB, previous_rgb: RGB,
            residual_density: Vector, vial_index: int, tolerance: float = 5.0,
            max_fraction: float = 0.5) -> dict:
    """Fit a constrained absorbance mixing curve; evidence, not causal proof."""
    observed, previous_rgb = rgb(observed), rgb(previous_rgb)
    tolerance = number(tolerance, "tolerance", 1e-9)
    max_fraction = number(max_fraction, "max_fraction")
    if max_fraction > 1 or isinstance(vial_index, bool) or not isinstance(vial_index, int) or vial_index < 1:
        raise ValueError("fraction must be <= 1 and vial_index a positive integer")
    clean_density = model.expected_density(recipe)
    clean = from_density(clean_density)
    # LAB distance along a nonlinear absorbance curve; grid resolution <= 0.001.
    candidates = [(delta_e(observed, from_density(blend(clean_density, residual_density, a))), a)
                  for a in (max_fraction*i/1000 for i in range(1001))]
    fit_error, fraction = min(candidates)
    clean_error = delta_e(observed, clean)
    separation = delta_e(clean, from_density(residual_density))
    improvement = (clean_error-fit_error)/max(clean_error, 1e-9)
    defect = clean_error > tolerance
    identifiable = separation > 2*tolerance
    likely = (defect and identifiable and vial_index == 1 and fraction > 0
              and fit_error <= tolerance and improvement >= 0.5)
    status = ("within_tolerance" if not defect else "carryover_consistent" if likely
              else "ambiguous_similar_colors" if not identifiable else "unexplained_color_defect")
    return {"status": status, "color_defect": defect,
            "expected_rgb": clean, "observed_rgb": observed, "previous_recipe_rgb": previous_rgb,
            "delta_e_expected": clean_error, "delta_e_previous": delta_e(observed, previous_rgb),
            "residual_rgb": from_density(residual_density), "fitted_residual_fraction": fraction,
            "fit_delta_e": fit_error, "relative_fit_improvement": improvement,
            "identifiable": identifiable, "fit_at_limit": fraction == max_fraction,
            "note": "Consistency score only; verify timing, later vials, lighting and dosing logs."}


def transition_risk(model: MixingModel, previous: Recipe, current: Recipe,
                    hold_up_ml: float, tolerance: float = 5.0) -> dict:
    tolerance = number(tolerance, "tolerance", 1e-9)
    tail = model.settled_tail(previous, hold_up_ml)
    predicted, _ = model.dispense(current, tail, hold_up_ml)
    difference = delta_e(predicted, model.expected_rgb(current))
    return {"score_0_100": 100*difference/(difference+tolerance),
            "warning": difference > tolerance, "predicted_first_rgb": predicted,
            "predicted_delta_e": difference, "hold_up_ml": hold_up_ml,
            "note": "Severity index, not probability; previous recipe assumed settled."}


def evaluate(labels: Sequence[str], predictions: Sequence[dict]) -> dict:
    """Ground truth labels: clean, carryover, other. Split data by run/transition."""
    if not labels or len(labels) != len(predictions) or set(labels)-{"clean", "carryover", "other"}:
        raise ValueError("Provide equal nonempty labels/predictions; labels: clean, carryover, other")
    def metrics(truth, predicted):
        tp = sum(t and p for t, p in zip(truth, predicted))
        fp = sum(not t and p for t, p in zip(truth, predicted))
        fn = sum(t and not p for t, p in zip(truth, predicted))
        tn = sum(not t and not p for t, p in zip(truth, predicted))
        ratio = lambda a, b: a/b if b else None
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": ratio(tp, tp+fp), "recall": ratio(tp, tp+fn),
                "f1": ratio(2*tp, 2*tp+fp+fn), "false_positive_rate": ratio(fp, fp+tn)}
    return {"defect_detection": metrics([x != "clean" for x in labels],
                                        [p["color_defect"] for p in predictions]),
            "carryover_explanation": metrics([x == "carryover" for x in labels],
                                             [p["status"] == "carryover_consistent" for p in predictions])}


def analyze(data: dict) -> dict:
    model = MixingModel(data.get("tank_rgb", DEMO_COLORS))
    previous = Recipe(**data["previous"])
    current = Recipe(**data["current"])
    h = number(data.get("hold_up_ml", 2), "hold_up_ml")
    tolerance = number(data.get("tolerance_delta_e", 5), "tolerance_delta_e", 1e-9)
    state = model.settled_tail(previous, h)
    previous_color = rgb(data.get("previous_last_rgb", model.expected_rgb(previous)))
    results = []
    for index, observation in enumerate(data.get("observations_rgb", []), 1):
        predicted, next_state = model.dispense(current, state, h)
        result = explain(model, current, rgb(observation), previous_color, state, index, tolerance)
        result.update(vial_index=index, hydraulic_predicted_rgb=predicted,
                      hydraulic_prediction_delta_e=delta_e(rgb(observation), predicted))
        results.append(result)
        state = next_state
    return {"synthetic_calibration": "tank_rgb" not in data,
            "dispense_order": [t for t in ORDER if current.volumes_ml.get(t, 0)],
            "expected_rgb": model.expected_rgb(current),
            "transition": transition_risk(model, previous, current, h, tolerance), "vials": results}


def demo() -> dict:
    model = MixingModel(DEMO_COLORS)
    blue = Recipe("Blue", {"T-400": 10})
    orange = Recipe("Orange", {"T-200": 10})
    clear = Recipe("Clear", {"T-100": 10})
    diluted = Recipe("Diluted blue", {"T-400": 5, "T-100": 5})
    state = model.settled_tail(blue, 2)
    observations = []
    for _ in range(3):
        observation, state = model.dispense(orange, state, 2)
        observations.append(observation)
    return {"warning": "SYNTHETIC demonstration; observations generated by the same model.",
            "examples": {r.name: model.expected_rgb(r) for r in (blue, orange, clear, diluted)},
            "blue_to_clear": transition_risk(model, blue, clear, 2),
            "blue_to_orange": analyze({"previous": {"name": blue.name, "volumes_ml": blue.volumes_ml},
                "current": {"name": orange.name, "volumes_ml": orange.volumes_ml},
                "hold_up_ml": 2, "observations_rgb": observations})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="JSON inspection input; omit for synthetic worked examples")
    args = parser.parse_args()
    try:
        if args.input:
            with open(args.input, encoding="utf-8-sig") as source:
                result = analyze(json.load(source))
        else:
            result = demo()
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

