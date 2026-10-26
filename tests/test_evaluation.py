import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import carryover
import evaluate_carryover as ev


def sequence():
    # Four independent mixture recipes; observations defined analytically, without
    # using the carryover simulator. This is a software fixture, not factory data.
    mixtures = [(20, 10, 0), (20, 0, 10), (10, 10, 10), (30, 10, 10)]
    seq = []
    for rid, volumes in enumerate(mixtures, 1):
        fractions = np.array(volumes) / sum(volumes)
        absorbance = np.array([.05, .03, .04]) + fractions[1] * np.array([.1, .5, .9]) + fractions[2] * np.array([.8, .4, .1])
        for pos in range(3):
            vial = len(seq) + 1
            seq.append(dict(vial=vial, recipe=rid, vols=dict(zip(carryover.TANKS, volumes)),
                            total=sum(volumes), pos=pos, obs=absorbance.copy(), fill=vial,
                            label='clean', calibration_clean=pos == 2, peak_frame=''))
    return seq


class EvaluationTests(unittest.TestCase):
    def test_held_out_observations_cannot_change_fitted_model(self):
        seq = sequence()
        original = ev.fit_fold(seq, 2, np.array([.5, 1.0]))
        for row in seq:
            if row['recipe'] == 2:
                row['obs'][:] = 999
        altered = ev.fit_fold(seq, 2, np.array([.5, 1.0]))
        np.testing.assert_array_equal(original['coef'], altered['coef'])
        self.assertEqual(original['rms'], altered['rms'])
        self.assertEqual(original['hold_up_ml'], altered['hold_up_ml'])
        self.assertEqual(original['well_mixed'], altered['well_mixed'])
        self.assertTrue(set(original['training_vials']).isdisjoint({4, 5, 6}))

    def test_noisy_or_unidentifiable_folds_are_not_silently_scored(self):
        seq = sequence()
        for row in seq:
            row['calibration_clean'] = False
        report = ev.evaluate(seq, np.array([.5]))
        self.assertEqual(report['evaluated_vials'], 0)
        self.assertIsNone(report['baseline_mean_error'])
        self.assertTrue(all(f['status'] == 'skipped' for f in report['folds']))

    def test_metrics_separate_cause_from_detection_and_skip_unknown(self):
        rows = [dict(label='carryover', defect=True, status='carryover_consistent'),
                dict(label='other', defect=True, status='carryover_consistent'),
                dict(label='clean', defect=False, status='within_tolerance'),
                dict(label='unknown', defect=True, status='carryover_consistent')]
        self.assertEqual(ev.metrics(rows)['precision'], 1)
        self.assertEqual(ev.metrics(rows, True)['precision'], .5)
        self.assertEqual(ev.metrics(rows)['n'], 3)
        self.assertIsNone(ev.metrics([])['recall'])

    def write_inputs(self, root, missing=False):
        seq = sequence()
        recipes, fills, alignment = [], [], []
        for s in seq:
            recipes.append(dict(vial=s['vial'], recipe_id=s['recipe'],
                                **dict(zip(('t100_ml', 't200_ml', 't400_ml'), s['vols'].values()))))
            absent = missing and s['vial'] == 2
            fill = s['vial'] - (1 if missing and s['vial'] > 2 else 0)
            if not absent:
                fills.append(dict(fill=fill, **dict(zip(('a_r', 'a_g', 'a_b'), s['obs'])), peak_frame='<script>bad</script>'))
            alignment.append(dict(vial=s['vial'], fill='' if absent else fill,
                                  status='missing' if absent else 'matched', label='unknown' if absent else 'clean',
                                  calibration_clean='1' if s['calibration_clean'] else '', evidence='synthetic fixture'))
        paths = [root / name for name in ('fills.csv', 'recipes.csv', 'alignment.csv')]
        for path, rows in zip(paths, (fills, recipes, alignment)):
            ev.write_csv(path, rows, list(rows[0]))
        return paths

    def test_missing_detection_keeps_later_recipe_identity_and_history(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self.write_inputs(Path(d), missing=True)
            seq, exclusions, unused = ev.load_aligned(*paths)
            self.assertEqual(len(seq), 12)
            self.assertIsNone(seq[1]['obs'])
            self.assertEqual((seq[2]['vial'], seq[2]['fill']), (3, 2))
            self.assertEqual((seq[3]['vial'], seq[3]['recipe']), (4, 2))
            self.assertEqual(len(exclusions), 1)
            self.assertEqual(unused, [])

    def test_duplicate_and_unreviewed_matches_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self.write_inputs(Path(d))
            rows = ev.read_csv(paths[2]); rows[1]['fill'] = rows[0]['fill']
            ev.write_csv(paths[2], rows, ev.FIELDS)
            with self.assertRaisesRegex(ValueError, 'reused'):
                ev.load_aligned(*paths)
            rows[1]['fill'] = '2'; rows[1]['evidence'] = ''
            ev.write_csv(paths[2], rows, ev.FIELDS)
            with self.assertRaisesRegex(ValueError, 'evidence'):
                ev.load_aligned(*paths)

    def test_carryover_loader_uses_reviewed_ids(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fills, recipes, alignment = self.write_inputs(root, missing=True)
            (root / 'labels').mkdir()
            (root / 'fills_video5.csv').write_bytes(fills.read_bytes())
            (root / 'labels/video5_vial_recipes.csv').write_bytes(recipes.read_bytes())
            with patch.object(carryover, 'OUT', str(root)), patch.object(carryover, 'VF', str(root)):
                seq = carryover.load(alignment)
            self.assertIsNone(seq[1]['obs'])
            self.assertEqual(seq[2]['fill'], 2)
            model = carryover.fit(seq, False, np.array([.5, 1.0]))
            self.assertTrue(np.isfinite(model['coef']).all())

    def test_missing_recipes_and_nonfinite_observations_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self.write_inputs(Path(d))
            rows = ev.read_csv(paths[0]); rows[0]['a_r'] = 'nan'
            ev.write_csv(paths[0], rows, list(rows[0]))
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                ev.load_aligned(*paths)
            mapping = ev.read_csv(paths[2])[:-1]
            ev.write_csv(paths[2], mapping, ev.FIELDS)
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                ev.load_aligned(*paths)

    def test_template_is_unreviewed_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); paths = self.write_inputs(root)
            target = root / 'review.csv'
            command = [sys.executable, str(Path(ev.__file__)), '--fills', str(paths[0]),
                       '--recipes', str(paths[1]), '--alignment', str(target), '--init-alignment']
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertTrue(target.with_suffix('.html').exists())
            with self.assertRaisesRegex(ValueError, 'review'):
                ev.load_aligned(paths[0], paths[1], target)
            repeated = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn('overwrite', repeated.stderr)

    def test_cli_report_and_real_data_guard(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); paths = self.write_inputs(root)
            command = [sys.executable, str(Path(ev.__file__)), '--fills', str(paths[0]),
                       '--recipes', str(paths[1]), '--alignment', str(paths[2]),
                       '--out', str(root / 'out'), '--grid-max', '1']
            denied = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(denied.returncode, 0)
            self.assertIn('organizer permission', denied.stderr)
            run = subprocess.run(command + ['--provenance', 'synthetic'], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads((root / 'out/summary.json').read_text())
            self.assertEqual(report['evaluated_vials'], 12)
            self.assertLess(report['baseline_mean_error'], 1e-10)
            page = (root / 'out/report.html').read_text()
            self.assertNotIn('<script>bad</script>', page)
            self.assertIn('&lt;script&gt;', page)
            self.assertIn('SYNTHETIC', page)


if __name__ == '__main__':
    unittest.main()
