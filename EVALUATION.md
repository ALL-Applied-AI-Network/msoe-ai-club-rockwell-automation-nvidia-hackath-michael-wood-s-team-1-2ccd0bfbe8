# Reviewed alignment and carryover evaluation

This contribution adds explicit vial identities, leave-one-recipe-out evaluation,
CSV/JSON results, and a self-contained HTML report. It reuses the physics model in
`src/carryover.py`. It does not require or merge the separate PartTwo branch.

## Why alignment changed

A detection number is not a vial ID. If vial 19 is missed, detected fill 19 may be
vial 20. Pairing by row number corrupts every later comparison. The previous code
worked around this by discarding all observations after vial 18.

The new workflow requires a reviewed CSV mapping. Missing/excluded vials keep their
recipe and place in the hydraulic simulation but have no observation to score.
Later verified matches can still be evaluated. This fixes identity handling; it
does **not** recover missed detections or claim an automatically verified mapping.
No real video-5 alignment is supplied: those identities still need footage review.

## Quick start: reproducible software example

From the repository root, with Python 3.10 or newer:

```sh
python -m pip install -r requirements-evaluation.txt
python -m unittest discover -s tests -v
python src/evaluate_carryover.py --fills examples/evaluation/fills.csv --recipes examples/evaluation/recipes.csv --alignment examples/evaluation/alignment.csv --provenance synthetic --grid-max 2 --out out/example-evaluation
```

Open `out/example-evaluation/report.html`. The example contains analytically
constructed colors with no carryover and one deliberately missing detection. It
is a software fixture, **not** digital-twin rendering or evidence of factory accuracy.
A baseline that already explains these observations should not gain from carryover.

## Review actual detected fills

First use the existing vision pipeline to generate `out/fills_video5.csv`, including
`a_r`, `a_g`, `a_b`. Obtain data through the event's shared folder; do not commit it.
Then create a blank mapping and an HTML review page:

```sh
python src/evaluate_carryover.py --fills out/fills_video5.csv --recipes data/vial_fill/labels/video5_vial_recipes.csv --alignment out/alignment.csv --init-alignment --images-dir data/vial_fill/images
```

The review page lists recipe times and detected-fill times/frames in independent
tables. Images are embedded when available. Review the video and timestamp offsets;
do not infer identity by color agreement, since that would bias color evaluation.
Fill in each CSV row:

| Column | Meaning |
|---|---|
| vial | Original labelled vial ID; exactly one row per vial |
| fill | Detected fill ID; blank for missing/excluded vials |
| status | matched, missing, or excluded |
| label | clean, carryover, other, or unknown |
| calibration_clean | 1 only for independently reviewed clean calibration samples; otherwise blank/0 |
| evidence | Required review notes, e.g. video timestamps and correspondence evidence |

Unknown cause labels are valid and excluded from classification metrics. Batch
position alone is not ground truth. Establish labels independently of model outputs.
If uncertain, exclude the observation and explain why. Duplicated, reversed, unknown
and unreviewed matches are rejected. Extra unmatched detections are reported.

## Evaluation and event data policy

The event expects digital-twin training data and separate real testing. The
leave-one-recipe-out command **fits coefficients and hold-up volume**. Real-data
cross-validation is an exploratory calibration experiment, not compliance with
synthetic-only training. Confirm organizer permission before using that experiment:

```sh
python src/evaluate_carryover.py --fills out/fills_video5.csv --recipes data/vial_fill/labels/video5_vial_recipes.csv --alignment out/alignment.csv --provenance real --allow-real-calibration --images-dir data/vial_fill/images
```

For actual synthetic observations use `--provenance synthetic`. Provenance is a
user declaration, not automatically verified. This PR does not implement Isaac Sim
rendering, a frozen synthetic-trained deployment model, or live hardware testing.

Each fold holds out **all observations of one recipe** from coefficient fitting,
hold-up selection and hydraulic-model selection. The full known recipe sequence
remains available as an input to the hydraulic simulation. Clean calibration rows
must identify a full-rank three-column design; unsupported folds are skipped with
reasons. No held-out colors are used to choose parameters.

This is an offline within-run recipe-generalization experiment. It can use training
observations later in the run, and is not a prospective online or held-out-run test.
The vision reference/geometry and thresholds are not refitted inside each fold.
Do not describe the results as an end-to-end unbiased benchmark. Hold thresholds
fixed before testing; use independent runs for a final system evaluation.

## Outputs and interpretation

- `summary.json`: fold memberships, parameters, coverage, skipped folds, exclusions,
  unmatched detections, input SHA-256 hashes, provenance, and elapsed evaluation time.
- `predictions.csv`: per-vial errors, reviewed labels, and assessments.
- `report.html`: observed/recipe-only/carryover color swatches, optional source frames,
  results, and complete metrics/provenance. Source frames are embedded up to 2 MB each.

Errors are Euclidean RGB **absorbance** distances, not Delta E or percentages.
The baseline and carryover predictions share training-only dye coefficients.
A color defect exceeds `--tolerance` (default 0.25). A carryover-consistent result
also needs error within tolerance and at least 30% improvement. These are provisional
thresholds, not calibrated probabilities or proof of causation.

Detection and carryover explanation get separate confusion matrices, precision,
recall, F1, and false-positive rates on independently labelled, evaluated samples.
Undefined metrics are null. Missing/excluded vials are shown in coverage, not counted
as correctly classified negatives. Detection coverage is distinct from classifier
recall. End-to-end missed-vial detection accuracy is not measured by these metrics.
Elapsed time is batch evaluation time, not camera inference latency.

`src/carryover.py` now requires `--alignment out/alignment.csv` and uses reviewed clean
calibration samples, allowing verified IDs beyond 18. Its normal fit remains an
in-sample exploratory report. `calibrate_tanks.py`, `run_integration.py`, and
`validate_vials.py` retain their older mapping assumptions; do not use their output
as this workflow's verified evaluation. The new evaluator consumes measurements
from `segment_fills.py` directly and requires reviewed alignment.

## Validation performed

Unit/integration tests exercise missed detections, duplicate/unreviewed identities,
held-out observation isolation, unidentifiable folds, separate cause metrics,
HTML escaping, the real-calibration guard, and CLI CSV/JSON/HTML generation.
Real-line accuracy and actual video-5 identity correction remain unverified.
