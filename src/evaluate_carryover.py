"""Reviewed vial alignment and leave-one-recipe-out evaluation. See EVALUATION.md."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import time

import numpy as np
import carryover

FIELDS = ['vial', 'fill', 'status', 'label', 'calibration_clean', 'evidence']


def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def keyed(rows, key):
    result = {}
    for row in rows:
        value = int(row[key])
        if value <= 0 or value in result:
            raise ValueError(f'Duplicate or nonpositive {key}: {value}')
        result[value] = row
    return result


def load_aligned(fills_path, recipes_path, alignment_path):
    """Never infer identity from detection order or observed colour.

    Every labelled vial requires a reviewed match or explicit exclusion. All recipes
    remain in simulation order, including vials whose observations are unavailable.
    """
    fills = keyed(read_csv(fills_path), 'fill')
    recipes = keyed(read_csv(recipes_path), 'vial')
    mapping = keyed(read_csv(alignment_path), 'vial')
    if not recipes or set(mapping) != set(recipes):
        raise ValueError('Alignment must contain exactly one row for every recipe vial')
    used, seq, exclusions = set(), [], []
    previous_recipe, position, previous_fill = None, -1, 0
    for vial, recipe in sorted(recipes.items()):
        row = mapping[vial]
        status = row.get('status', '')
        label = row.get('label', '') or 'unknown'
        if label not in ('unknown', 'clean', 'carryover', 'other'):
            raise ValueError(f'Invalid label for vial {vial}')
        if row.get('calibration_clean', '') not in ('', '0', '1'):
            raise ValueError('calibration_clean must be blank, 0 or 1')
        if status not in ('matched', 'missing', 'excluded') or not row.get('evidence', '').strip():
            raise ValueError(f'Vial {vial}: review status and evidence are required')
        observation, source, fill = None, {}, None
        if status == 'matched':
            fill = int(row['fill'])
            if fill not in fills or fill in used or fill <= previous_fill:
                raise ValueError(f'Vial {vial}: unknown, reused or out-of-order fill {fill}')
            used.add(fill)
            previous_fill = fill
            source = fills[fill]
            observation = np.array([float(source[k]) for k in ('a_r', 'a_g', 'a_b')])
            if not np.isfinite(observation).all():
                raise ValueError(f'Vial {vial}: nonfinite absorbance')
        else:
            if row.get('fill', '') or row.get('calibration_clean') == '1':
                raise ValueError('Missing/excluded vials cannot have a fill or calibration role')
            exclusions.append({'vial': vial, 'status': status, 'reason': row['evidence']})
        clean = row.get('calibration_clean') == '1'
        if clean and label != 'clean':
            raise ValueError('Calibration samples require an independently reviewed clean label')
        rid = int(recipe['recipe_id'])
        position = position + 1 if rid == previous_recipe else 0
        previous_recipe = rid
        volumes = {tank: float(recipe[key] or 0) for tank, key in
                   zip(carryover.TANKS, ('t100_ml', 't200_ml', 't400_ml'))}
        if not all(math.isfinite(v) and v >= 0 for v in volumes.values()) or sum(volumes.values()) <= 0:
            raise ValueError(f'Vial {vial}: invalid recipe volume')
        seq.append(dict(vial=vial, recipe=rid, vols=volumes, total=sum(volumes.values()),
                        pos=position, obs=observation, fill=fill, label=label,
                        calibration_clean=clean, evidence=row['evidence'],
                        peak_frame=source.get('peak_frame', '')))
    return seq, exclusions, sorted(set(fills) - used)


def fit_fold(seq, held_out, grid):
    """Fit coefficients and hydraulics without reading held-out observations."""
    clean = [s for s in seq if s['recipe'] != held_out and
             s['obs'] is not None and s['calibration_clean']]
    if len(clean) < 3:
        raise ValueError('fewer than three reviewed clean training observations')
    design = carryover.design(np.array([[s['vols'][t] / s['total'] for t in carryover.TANKS]
                                       for s in clean]))
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError('clean training recipes cannot identify both dyes and baseline')
    coef = np.linalg.lstsq(design, np.array([s['obs'] for s in clean]), rcond=None)[0]
    keep = [i for i, s in enumerate(seq) if s['recipe'] != held_out and s['obs'] is not None]
    observed = np.array([seq[i]['obs'] for i in keep])
    best = None
    for mixed in (False, True):
        for volume in grid:
            mixtures = carryover.simulate(seq, float(volume), mixed)[keep]
            rms = float(np.sqrt(np.mean((observed - carryover.design(mixtures) @ coef) ** 2)))
            if best is None or rms < best['rms']:
                best = dict(coef=coef, hold_up_ml=float(volume), well_mixed=mixed, rms=rms,
                            at_grid_edge=bool(volume in (grid[0], grid[-1])),
                            training_vials=[seq[i]['vial'] for i in keep])
    return best


def metrics(rows, explanation=False):
    labelled = [r for r in rows if r['label'] != 'unknown']
    counts = dict(tp=0, fp=0, fn=0, tn=0)
    for row in labelled:
        actual = row['label'] == 'carryover' if explanation else row['label'] != 'clean'
        predicted = row['status'] == 'carryover_consistent' if explanation else row['defect']
        counts['tp' if actual and predicted else 'fn' if actual else 'fp' if predicted else 'tn'] += 1
    tp, fp, fn, tn = (counts[k] for k in ('tp', 'fp', 'fn', 'tn'))
    def ratio(a, b):
        return a / b if b else None
    return dict(n=len(labelled), **counts, precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                f1=ratio(2 * tp, 2 * tp + fp + fn), false_positive_rate=ratio(fp, fp + tn))


def evaluate(seq, grid, tolerance=0.25, improvement=0.30):
    results, folds = [], []
    for rid in sorted({s['recipe'] for s in seq}):
        test = [i for i, s in enumerate(seq) if s['recipe'] == rid and s['obs'] is not None]
        if not test:
            folds.append(dict(recipe=rid, status='skipped', reason='no matched observations'))
            continue
        try:
            model = fit_fold(seq, rid, grid)
        except ValueError as exc:
            folds.append(dict(recipe=rid, status='skipped', reason=str(exc)))
            continue
        baseline, _ = carryover.predict(seq, model, clean=True)
        adjusted, _ = carryover.predict(seq, model, clean=False)
        folds.append(dict(recipe=rid, status='evaluated', test_vials=[seq[i]['vial'] for i in test],
                          **{k: v for k, v in model.items() if k != 'coef'}))
        for i in test:
            s = seq[i]
            ec = float(np.linalg.norm(s['obs'] - baseline[i]))
            ed = float(np.linalg.norm(s['obs'] - adjusted[i]))
            defect = ec > tolerance
            consistent = defect and ed <= tolerance and ed <= ec * (1 - improvement)
            status = 'within_tolerance' if not defect else 'carryover_consistent' if consistent else 'unexplained_color_defect'
            results.append(dict(vial=s['vial'], recipe=rid, fill=s['fill'], label=s['label'],
                                baseline_error=ec, carryover_error=ed, defect=defect, status=status,
                                observed=s['obs'].tolist(), expected=baseline[i].tolist(),
                                with_carryover=adjusted[i].tolist(), peak_frame=s['peak_frame']))
    results.sort(key=lambda r: r['vial'])
    mean = lambda key: float(np.mean([r[key] for r in results])) if results else None
    return dict(rows=results, folds=folds, baseline_mean_error=mean('baseline_error'),
                carryover_mean_error=mean('carryover_error'), defect_detection=metrics(results),
                carryover_explanation=metrics(results, True), evaluated_vials=len(results),
                expected_vials=len(seq), matched_vials=sum(s['obs'] is not None for s in seq))


def swatch(absorbance):
    linear = np.clip(10.0 ** -np.maximum(absorbance, 0), 0, 1)
    rgb = np.rint(255 * np.where(linear <= 0.0031308, 12.92 * linear,
                                1.055 * linear ** (1 / 2.4) - 0.055)).astype(int)
    color = '#%02x%02x%02x' % tuple(rgb)
    return f'<span class="swatch" style="background:{color}"></span> {color}'


def image_cell(filename, images_dir):
    if images_dir is None or not filename:
        return html.escape(filename)
    root = Path(images_dir).resolve()
    path = (root / filename).resolve()
    if not path.is_relative_to(root) or path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
        return html.escape(filename) + ' (unsupported image path)'
    if not path.is_file() or path.stat().st_size > 2_000_000:
        return html.escape(filename) + ' (image unavailable or over 2 MB)'
    mime = 'image/png' if path.suffix.lower() == '.png' else 'image/jpeg'
    data = base64.b64encode(path.read_bytes()).decode('ascii')
    return f'<img width="240" alt="{html.escape(filename, quote=True)}" src="data:{mime};base64,{data}">'


def alignment_review(fills, recipes, images_dir=None):
    def table(rows):
        if not rows:
            return '<p>No rows</p>'
        return '<table><tr>' + ''.join('<th>'+html.escape(k)+'</th>' for k in rows[0]) + '</tr>' + ''.join(
            '<tr>'+''.join('<td>'+str(v)+'</td>' for v in row.values())+'</tr>' for row in rows) + '</table>'
    recipe_rows = [{k: html.escape(str(v)) for k, v in r.items()} for r in read_csv(recipes)]
    fill_rows = [{k: image_cell(v, images_dir) if k == 'peak_frame' else html.escape(str(v))
                  for k, v in r.items()} for r in read_csv(fills)]
    return ('<!doctype html><meta charset="utf-8"><title>Alignment review</title>'
            '<style>body{font:15px system-ui;margin:25px}td,th{padding:8px;border:1px solid #aaa}table{border-collapse:collapse}</style>'
            '<h1>Review vial identities before evaluation</h1><p>Tables are independent. '
            'Do not pair by row number or expected colour. Verify video timing and sequence; '
            'record clock offsets and evidence in the alignment CSV. Mark uncertain vials excluded.</p>'
            '<h2>Labelled recipes</h2>' + table(recipe_rows) + '<h2>Detected fills</h2>' + table(fill_rows))


def report_html(report, images_dir=None):
    esc = html.escape
    body = []
    for row in report['rows']:
        body.append('<tr>' + ''.join(f'<td>{value}</td>' for value in
                    [row['vial'], row['recipe'], swatch(row['observed']), swatch(row['expected']),
                     swatch(row['with_carryover']), f"{row['baseline_error']:.3f}",
                     f"{row['carryover_error']:.3f}", esc(row['status']), esc(row['label']),
                     image_cell(row['peak_frame'], images_dir)]) + '</tr>')
    summary = {k: v for k, v in report.items() if k != 'rows'}
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Carryover evaluation</title>
<style>body{font:16px system-ui;margin:32px;color:#17243a}table{border-collapse:collapse;width:100%}
th,td{text-align:left;padding:10px;border-bottom:1px solid #ccd4df}.swatch{display:inline-block;width:28px;height:22px;vertical-align:middle;border:1px solid #555}pre{white-space:pre-wrap;background:#f0f3f8;padding:20px}</style>
<h1>Recipe-aware carryover evaluation</h1><p>''' + esc(report['notice']) + '''</p>
<p>Errors are Euclidean RGB absorbance distances, not perceptual Delta E. Swatches are illustrative.
All exclusions and skipped folds are included below. Cause labels require independent review.</p>
<table><thead><tr>''' + ''.join(f'<th>{x}</th>' for x in
['Vial', 'Recipe', 'Observed', 'Recipe only', 'With carryover', 'Baseline error', 'Carryover error',
 'Assessment', 'Reviewed label', 'Source frame']) + '</tr></thead><tbody>' + ''.join(body) + \
'</tbody></table><h2>Coverage, metrics and provenance</h2><pre>' + esc(json.dumps(summary, indent=2)) + '</pre></html>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fills', type=Path, required=True)
    parser.add_argument('--recipes', type=Path, required=True)
    parser.add_argument('--alignment', type=Path, required=True)
    parser.add_argument('--images-dir', type=Path, help='optional vial_fill/images directory to embed source frames')
    parser.add_argument('--init-alignment', action='store_true')
    parser.add_argument('--out', type=Path, default=Path('out/evaluation'))
    parser.add_argument('--provenance', choices=['synthetic', 'real'], default='real')
    parser.add_argument('--allow-real-calibration', action='store_true')
    parser.add_argument('--tolerance', type=float, default=0.25)
    parser.add_argument('--grid-max', type=float, default=80)
    parser.add_argument('--grid-step', type=float, default=0.5)
    args = parser.parse_args()
    try:
        if args.init_alignment:
            if args.alignment.exists():
                raise ValueError('Refusing to overwrite an existing alignment')
            recipes = keyed(read_csv(args.recipes), 'vial')
            args.alignment.parent.mkdir(parents=True, exist_ok=True)
            write_csv(args.alignment, [dict(vial=v, fill='', status='', label='unknown',
                                            calibration_clean='', evidence='') for v in sorted(recipes)], FIELDS)
            args.alignment.with_suffix('.html').write_text(alignment_review(args.fills, args.recipes, args.images_dir), encoding='utf-8')
            print(f'Created {args.alignment} and review HTML; review all rows before evaluation.')
            return
        if args.provenance == 'real' and not args.allow_real_calibration:
            raise ValueError('This experiment fits real observations. Confirm organizer permission, then use --allow-real-calibration; synthetic training is the event expectation.')
        if not all(math.isfinite(v) and v > 0 for v in (args.tolerance, args.grid_max, args.grid_step)) or args.grid_step > args.grid_max:
            raise ValueError('Tolerance and grid bounds must be finite and positive; step <= max')
        started = time.perf_counter()
        seq, exclusions, unmatched = load_aligned(args.fills, args.recipes, args.alignment)
        grid = np.arange(args.grid_step, args.grid_max + 1e-9, args.grid_step)
        report = evaluate(seq, grid, args.tolerance)
        report.update(exclusions=exclusions, unmatched_fills=unmatched, provenance=args.provenance,
                      tolerance=args.tolerance, minimum_relative_improvement=0.30,
                      grid=grid.tolist(), elapsed_seconds=time.perf_counter() - started,
                      input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                    (args.fills, args.recipes, args.alignment)},
                      notice=('SYNTHETIC validation: not evidence of real-line accuracy.' if args.provenance == 'synthetic'
                              else 'Exploratory real-data calibration: not a synthetic-trained benchmark.'))
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        fields = ['vial', 'recipe', 'fill', 'label', 'baseline_error', 'carryover_error', 'defect', 'status', 'peak_frame']
        write_csv(args.out / 'predictions.csv', [{k: r[k] for k in fields} for r in report['rows']], fields)
        (args.out / 'report.html').write_text(report_html(report, args.images_dir), encoding='utf-8')
        print(f"Evaluated {report['evaluated_vials']}/{len(seq)} vials; report: {args.out / 'report.html'}")
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
