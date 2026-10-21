# Fluid-mixing inspection prototype

Run with Python 3.10 or later; no packages required:

```powershell
python -B RecipeAnalysis.py
python -B RecipeAnalysis.py --input recipe_example.json
python -B -m unittest discover -s tests -v
```

The default demonstration prints JSON with expected colors, transition severity,
three simulated vial observations, and explanations. All default tank colors,
hold-up volumes, and thresholds are illustrative. Synthetic observations are
created with the prediction model and cannot establish real-world accuracy.

## 1. Recipe Understanding

T-100 is clear diluent, T-200 orange, T-400 blue. T-300 is out of service:
the program rejects any reference to it, including a zero-volume entry. Unknown
tanks, negative/nonfinite volumes, and empty recipes are rejected.

Each recipe specifies volumes **per vial**. If station recipes instead describe
an entire batch, divide by the number of equal-volume vials before analysis.
The total vial volume is V = sum(v_i), and each fraction is f_i = v_i / V.
Dispensing is always T-200 -> T-400 -> T-100, skipping zero/absent entries.

Example: 2 mL orange + 3 mL blue + 5 mL clear gives fractions (0.2, 0.3, 0.5)
and a 10 mL vial. Clear changes concentration, rather than adding a white pigment.
Use the last vial as an empirical clean reference only after confirming stability
across later vials. It is a useful project assumption, not an automatic guarantee.

## 2. Expected Color Modeling

Use **camera RGB for acquisition**, approximate **optical density for physical
mixing**, and **CIELAB for inspection distance**. HSV is useful for a simple
operator display, but hue wraps at 360 degrees and becomes unstable near clear
or gray samples. Averaging hue or gamma-encoded RGB is unsuitable for predicting
absorbing liquids. LAB is a comparison space, not a physical mixing law.

For monochromatic transmission, Beer-Lambert gives A(lambda) = epsilon(lambda)
* concentration * path_length, and transmission T = 10^(-A). Absorbances of
independent absorbers add. The prototype approximates this using three broad
camera channels:

```text
s = RGB_channel / 255
linear(s) = s/12.92                         if s <= 0.04045
            ((s + 0.055)/1.055)^2.4        otherwise
A_i,c = -log10(max(0.000001, linear(s_i,c)))
A_recipe,c = sum(f_i * A_i,c)
T_recipe,c = 10^(-A_recipe,c)
RGB_recipe = inverse_sRGB_transfer(T_recipe)
```

Assumptions: fixed vial geometry and optical path, locked exposure/white balance,
white backlighting, comparable fluid levels, no chemical reaction or scattering,
and calibration samples captured with the same optics. Camera channels integrate
spectra, so channel-wise absorbance is only an approximation to spectral mixing.
Dark offsets and a measured clear reference should be corrected in a real camera
pipeline before this model; the prototype assumes white-referenced sRGB samples.
Zero channels are floored, preventing infinities but limiting very dark predictions.

Worked examples using the synthetic samples:

| Recipe | Expected approximate RGB |
| --- | --- |
| 10 mL orange | (240, 125, 30) |
| 10 mL blue | (35, 100, 225) |
| 10 mL clear | (255, 255, 255) |
| 5 mL blue + 5 mL clear | (101, 161, 240) |

For the last row, A_mix = 0.5 A_blue, so each channel transmission is
sqrt(T_blue). Orange dilution follows the same formula; orange plus blue can
become dark/desaturated through subtractive absorption, rather than a bright
midpoint between display RGB values.

The code converts sRGB to XYZ with D65 white and then LAB. It compares samples
with DeltaE76 = sqrt((L1-L2)^2 + (a1-a2)^2 + (b1-b2)^2). This is easy to audit;
its perceptual uniformity is imperfect. Consider CIEDE2000 later if useful, and
calibrate any tolerance from station observations rather than human visibility
rules. A synthetic default tolerance of 5 is used throughout.

Advantages: interpretable, inexpensive, works before a large training dataset.
Limitations: shadows, bubbles, reflections, colored background, turbidity, nonlinear
camera processing and different liquid depths can all invalidate the approximation.
A practical improvement is an empirical regression from recipe fractions to LAB,
trained on verified clean last vials and evaluated on unseen recipe compositions.
Compare it against this physical baseline before choosing the more complex model.

## 3. Carryover Modeling

Treat the shared path as a perfectly stirred fixed-volume chamber of hold-up H.
At each tank's input, integrate in dispensed volume q:

```text
dA_path/dq = (A_in - A_path)/H
A_end = A_in + (A_start - A_in) * exp(-v/H)
Integrated outlet density = v*A_in - H*(A_end - A_start)
A_collected = sum(integrated outlet density for each tank) / V
```

The collected vial is assumed fully mixed before imaging. The code executes each
tank in dispensing order and retains the final path state. At H=0 it returns the
recipe mixture exactly. A plug-flow alternative would use alpha = min(H/V, 1)
for a uniform inlet; neither transport model should be assumed correct without
measurements. Real piping may contain delay, stagnant pockets and dispersion.

For a uniform inlet, average residual fraction in vial k is:

```text
alpha_k = (H/V)*(1-exp(-V/H))*exp(-(k-1)*V/H)
A_vial,k = (1-alpha_k)*A_current + alpha_k*A_previous_path
```

Worked example: blue -> orange, H=2 mL, V=10 mL. First-vial alpha is 0.19865,
second is 0.00134, third is 0.00000902. Thus the first orange vial has about
19.9% blue-path contribution in this model; later vials approach clean orange.
Blue -> clear uses the same fractions and can remain visibly tinted. Clear ->
blue dilutes the first blue vial. These are volume contributions, not perceptual
color percentages.

**The previous path is not necessarily the previous vial color.** With a recipe
ending in a large clear draw, the final path may be mostly clear while the vial
remains blue/orange. The code analytically solves the periodic path state after
repeated previous-recipe fills, then predicts the next transition. This assumes
the previous recipe was already settled and there was no intervening flush.
For a short batch, use `dispense` with the actual tracked path state instead.

Estimate H by minimizing sum(DeltaE(predicted_k(H), observed_k)^2) on recorded
transitions with known initial state. Evaluate the fitted H on held-out runs;
compare stirred-path and plug-flow predictions. The current CLI accepts H as a
parameter and does not claim to estimate it from synthetic data.

## 4. Root Cause Explanation Logic

First flag a color defect when d_clean = DeltaE(observed, expected) > tolerance.
Then fit a constrained one-parameter residual mixing curve:

```text
A_fit(alpha) = (1-alpha)*A_clean + alpha*A_previous_path
alpha_hat = argmin_[0, alpha_max] DeltaE(observed, RGB(A_fit(alpha)))
improvement = (d_clean - d_fit) / max(d_clean, 1e-9)
```

The prototype searches 1,001 points up to alpha_max=0.5. It reports
`carryover_consistent` only for vial 1, with a color defect, fit error <= tolerance,
at least 50% improvement, positive fitted alpha, and residual/clean separation
exceeding twice the tolerance. It also reports distance to the previous recipe's
last-vial color (or a predicted previous color if no measured reference is given).
The path color supplies the physical residual hypothesis; previous-vial distance
provides operator context, not an independent causal test.

Worked example: synthetic blue -> orange at H=2 and V=10 has fitted alpha near
0.199 and nearly zero fit error, while the clean-orange error is appreciable.
It is consistent with carryover. An unrelated abnormal color far from the blue/
orange absorbance curve remains `unexplained_color_defect`. Blue -> blue has
little color separation, so color cannot reliably identify carryover. The output
explicitly reports identifiability and whether the fit reached its upper bound.

The residual curve is a simpler explanation model than the sequential hydraulic
prediction. Multiple sequential ingredients can produce a transient that is not
on that curve. For this reason the output separately includes the full hydraulic
prediction error; a failed curve fit does not exclude carryover. Large H can also
contaminate later vials, although this conservative heuristic only attributes the
first vial. Use `dispense` and actual history to investigate those cases.

Advantages: transparent evidence and an explicit unknown outcome. Limitations:
wrong dosing, a stained vial, and changed lighting can mimic residual mixing.
Confidence improves when later vials recover and valve/flow logs support the
transition history. The prototype reports evidence, not a calibrated probability
or proof of causation. Do not force every defect into the carryover class.

## 5. Risk Prediction

Before filling, predict the first vial by applying the current dispense sequence
to the previous settled path state. Compute:

```text
d = DeltaE(predicted_first, expected_clean)
risk_index = 100*d/(d+tolerance)
warning = d > tolerance
```

This monotonic severity index is 0 for an exact match, 50 at the configured
tolerance, and approaches 100 for increasingly large deviations. It is not a
probability. A worked numerical example: d=10 and tolerance=5 gives 66.7;
d=2 gives 28.6. The demo prints actual model values for blue -> clear and
blue -> orange. Identical repeated recipes at periodic steady state score zero;
a blue -> clear transition can score high despite using only clear in the new
recipe. Increasing vial volume relative to H reduces residual fractions.

Advantages: needs recipes and a calibrated H, no camera frame at prediction time.
Limitations: assumes known path history and ignores uncertainty in pumps, valves,
and H. Run a range of plausible H values for a sensitivity band before deploying
warnings. A path ending in clear may have lower risk than its previous vial color
alone suggests. Optimize warnings for observed defect rate and operator workload.

## 6. Factory Workflow Integration

1. Before dispensing, validate tank availability and compute transition risk.
2. Display previous/current recipe, expected and predicted first-vial swatches,
   severity and a plain-language explanation such as "Blue residue may tint the
   first clear vial; inspect first vial under the approved procedure."
3. Capture an image after settling; verify fill level and image quality first.
4. Extract liquid-only median RGB from a fixed ROI, excluding glass edges,
   labels, meniscus and glare. This prototype accepts extracted RGB, not images.
5. Compare each vial and show color deviation plus explanation evidence.
6. Check later-vial recovery, retain the last stable vial as a calibration candidate,
   and log operator disposition with recipe IDs, timestamps and model version.

Example: blue -> clear triggers advance inspection; if only vial 1 is tinted and
later vials recover, record suspected carryover. If all orange vials stay wrong,
inspect dosing/calibration as competing explanations. Use the station's existing
quality and flushing procedures. This code does not command pumps, change recipes,
automatically reject product, or implement a safety interlock.

## 7. Prototype Architecture

```text
Recipe JSON + calibration + hold-up estimate
                  |
          recipe validation
                  |
     optical-density / path simulation -> advance warning
                  |
Camera -> quality gate -> liquid ROI RGB -> LAB comparison
                  |                         |
          later-vial reference       carryover curve fit
                  |                         |
                JSON report -> operator review + labeled run log
```

`RecipeAnalysis.py` contains the mathematical model, explanation heuristic,
transition scoring, simple evaluation metrics, and JSON CLI. `recipe_example.json`
is an editable example. `tests/test_recipe_analysis.py` verifies key invariants.
A student MVP needs one fixed camera/backlight, manually selected ROI, recipe
records, and these interpretable functions. A lightweight display can render the
JSON later; an LLM or GPU is not necessary for this baseline. Adding machine
learning is justified once labeled data establishes where the baseline fails.

Input fields: `previous` and `current` have `name` and `volumes_ml`. `tank_rgb`
optionally replaces all three synthetic calibration colors; `hold_up_ml` and
`tolerance_delta_e` are configurable. `observations_rgb` is a chronological list
of vial colors, starting with vial 1. `previous_last_rgb` optionally supplies a
measured previous reference. Missing observations still permits advance risk.
The synthetic calibration flag is false when colors are supplied; this only
indicates user-provided values, not independently verified measurements.

## 8. Recommended Next Steps

Collect clean last-vial repeats and labeled transitions: blue -> orange,
orange -> blue, both colors -> clear, clear -> each color, identical recipes,
and mixtures ending in clear. Vary vial volumes, collect at least three vials
per transition, and record flushing and initial path conditions. Include confirmed
wrong dosing and lighting/bubble defects so explanation testing has alternatives.

Fit calibration and H on training runs. Tune tolerance and fit criteria on separate
validation runs. Hold out entire runs/days and recipe transitions for final testing;
random splitting adjacent vials risks leakage. Do not use a test vial's last-vial
reference as an input to a claimed pre-fill prediction. Post-run analysis can use
it if labeled as retrospective.

Report the following, with counts and uncertainty intervals by run:

| Goal | Metrics |
| --- | --- |
| Expected-color accuracy | Mean/median/95th-percentile DeltaE on clean vials |
| Defect detection | Precision, recall, F1, false-positive rate, confusion matrix |
| Carryover explanation | Precision/recall/F1 against experimentally confirmed cause |
| Unknown handling | Abstention rate; accuracy among non-abstained explanations |
| Advance warning | Precision/recall at threshold; false warnings per 100 transitions |
| Operations | Missed first-vial defects, alert count, time to disposition |

`evaluate()` implements separate binary defect and carryover confusion metrics;
undefined ratios return null. It requires independent labels `clean`, `carryover`,
or `other`. Fitted alpha accuracy requires independent concentration/tracer
measurements; do not use the model's own alpha as ground truth. Probability
calibration metrics such as Brier score apply only after training and validating
a probability model, not directly to the severity index.

Example: among 20 independently confirmed carryover defects, 16 identified gives
80% explanation recall. If 4 other events are also attributed to carryover,
precision is 16/(16+4)=80%. Report detection and explanation separately: a system
can find a bad vial while giving the wrong reason. Synthetic demonstrations and
unit tests verify mechanics, not these performance figures.

Minimum competition deliverable: reproducible CLI demo, measured calibration,
recorded real transitions, held-out evaluation, and an operator-facing explanation
with clear unknown cases. Expand to camera automation after that baseline works.

References supporting the physical/color definitions (the thresholds and decision
heuristics here are project proposals):

- IUPAC Beer-Lambert law: https://goldbook.iupac.org/terms/view/B00626
- CIE CIELAB color difference: https://cie.co.at/eilvterm/17-23-077

