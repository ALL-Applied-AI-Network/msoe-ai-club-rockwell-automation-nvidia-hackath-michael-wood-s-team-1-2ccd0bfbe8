"""Mathematical invariants and meaningful positive/negative explanation cases."""
import math
import unittest
from RecipeAnalysis import (DEMO_COLORS, MixingModel, Recipe, analyze, delta_e,
                            evaluate, explain, optical_density, transition_risk)


class RecipeAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.model = MixingModel(DEMO_COLORS)
        self.blue = Recipe("blue", {"T-400": 10})
        self.orange = Recipe("orange", {"T-200": 10})

    def test_invalid_recipes(self):
        for volumes in ({"T-300": 0, "T-100": 10}, {"T-999": 1}, {},
                        {"T-100": -1}, {"T-100": math.nan}, {"T-100": True}):
            with self.subTest(volumes=volumes), self.assertRaises(ValueError):
                Recipe("invalid", volumes)

    def test_color_roundtrip_and_reference_lab(self):
        from RecipeAnalysis import from_density, lab
        for color in DEMO_COLORS.values():
            self.assertLess(delta_e(color, from_density(optical_density(color))), 1e-8)
        self.assertAlmostEqual(lab((255, 255, 255))[0], 100, places=4)
        self.assertAlmostEqual(lab((0, 0, 0))[0], 0)

    def test_no_hold_up_equals_recipe_mixture(self):
        recipe = Recipe("mixed", {"T-200": 2, "T-400": 3, "T-100": 5})
        observed, _ = self.model.dispense(recipe, optical_density(DEMO_COLORS["T-400"]), 0)
        self.assertLess(delta_e(observed, self.model.expected_rgb(recipe)), 1e-8)

    def test_mass_balance_and_exponential_tail(self):
        h, v = 2, 10
        initial = self.model.expected_density(self.blue)
        inlet = self.model.expected_density(self.orange)
        observed, tail = self.model.dispense(self.orange, initial, h)
        collected = optical_density(observed)
        for i in range(3):
            self.assertAlmostEqual(tail[i], inlet[i]+(initial[i]-inlet[i])*math.exp(-v/h))
            self.assertAlmostEqual(v*collected[i]+h*tail[i], v*inlet[i]+h*initial[i])

    def test_clear_last_changes_path_but_not_steady_vial_composition(self):
        recipe = Recipe("diluted", {"T-100": 20, "T-400": 10, "T-200": 5})
        tail = self.model.settled_tail(recipe, 2)
        output, end = self.model.dispense(recipe, tail, 2)
        self.assertLess(delta_e(output, self.model.expected_rgb(recipe)), 1e-8)
        self.assertLess(max(abs(a-b) for a, b in zip(tail, end)), 1e-10)
        self.assertLess(max(tail), 0.001)  # Clear is last despite dict insertion order.
        self.assertLess(transition_risk(self.model, recipe, recipe, 2)["score_0_100"], 1e-6)

    def test_first_vial_carryover_and_later_recovery(self):
        state = self.model.settled_tail(self.blue, 2)
        observations = []
        for _ in range(3):
            color, state = self.model.dispense(self.orange, state, 2)
            observations.append(color)
        report = analyze({"previous": {"name": "blue", "volumes_ml": {"T-400": 10}},
                          "current": {"name": "orange", "volumes_ml": {"T-200": 10}},
                          "observations_rgb": observations})
        first, second, third = report["vials"]
        self.assertEqual(first["status"], "carryover_consistent")
        self.assertAlmostEqual(first["fitted_residual_fraction"], 0.19865, delta=0.001)
        self.assertEqual(second["status"], "within_tolerance")
        self.assertLess(third["delta_e_expected"], second["delta_e_expected"])
        self.assertTrue(report["transition"]["warning"])

    def test_unrelated_color_not_attributed_to_carryover(self):
        result = explain(self.model, self.orange, (20, 240, 20), DEMO_COLORS["T-400"],
                         self.model.expected_density(self.blue), 1)
        self.assertEqual(result["status"], "unexplained_color_defect")

    def test_similar_colors_are_not_identifiable(self):
        result = explain(self.model, self.blue, (240, 20, 20), DEMO_COLORS["T-400"],
                         self.model.expected_density(self.blue), 1)
        self.assertFalse(result["identifiable"])
        self.assertEqual(result["status"], "ambiguous_similar_colors")

    def test_evaluation_keeps_detection_and_explanation_separate(self):
        result = evaluate(["carryover", "other", "clean"], [
            {"color_defect": True, "status": "carryover_consistent"},
            {"color_defect": True, "status": "carryover_consistent"},
            {"color_defect": False, "status": "within_tolerance"}])
        self.assertEqual(result["defect_detection"]["precision"], 1)
        self.assertEqual(result["carryover_explanation"]["precision"], 0.5)


if __name__ == "__main__":
    unittest.main()
