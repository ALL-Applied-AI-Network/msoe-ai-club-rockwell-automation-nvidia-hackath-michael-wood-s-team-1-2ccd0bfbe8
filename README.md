# Recipe-Aware Color Monitoring and Carryover Alerts

Rockwell Automation + NVIDIA Innovation Lab, Fall 2026 · Dataset 2: Fluid Mixing Station

Residual fluid from the previous recipe can discolour the first vial of the next batch, and the
existing inspection only checks that a colour is *known*, not that it matches the *intended*
recipe. This system watches the fill-station camera, says what colour each vial **should** be from
its recipe, flags vials that don't match, explains **why** (e.g. "residue from Recipe 6 in the
shared mixing path"), and warns about risky changeovers **before** they run.

## Run it at the line (no dataset needed)

Python 3.10 or newer.

**Windows:** double-click `run_dashboard.bat`. **Mac/Linux:** `./run_dashboard.sh`.
Or by hand:

```bash
python -m pip install -r requirements.txt
python src/dashboard.py --list-cameras     # saves out/camera_N.jpg for every camera found
python src/dashboard.py --source 0         # the number of the camera looking at the fill nozzle
```

The page opens at <http://127.0.0.1:8050>. Then:

1. **Camera.** Pick the source in the box at the top: a camera number, an `rtsp://` / `http://`
   stream, a video file, or a folder that images are saved into (new files are picked up live).
2. **Align.** Park an **empty** vial under the nozzle and press **Auto-align**. The YOLO vial
   locator finds the vial and the frame is shifted onto the measured geometry, so a camera that
   sits differently from the dataset camera still measures the right pixels. Check the yellow
   box sits on the vial; `dx` / `dy` can be set by hand.
3. **Reference.** With the empty vial still parked, press **Capture empty reference**. Colour is
   measured as absorbance against this frame, which cancels the local lighting and glass.
4. **Recipes.** Enter the recipes in the order they will run (or pick from the labelled ones).
   The *n*-th finished vial is judged against the *n*-th slot. Recipes outside the station limits
   (> 80 mL, T-200 > 50 mL, a used tank < 10 mL) are flagged red.
5. **Run.** Each finished vial appears with its photo, the detected colour, the expected colour,
   the colour predicted *with* carryover, a verdict (OK / CARRYOVER / MISMATCH) and the reason.
   The **changeover risk** table predicts the first vial of each upcoming recipe change.
6. **Score it.** Press *good* / *off-colour* on each vial; caught / missed / false alarms are
   tallied live. Everything is logged to `out/line_test/<session>/` (CSV + vial photos).

The AI vial locator runs from `models/vial_yolo.onnx` through OpenCV, so no PyTorch is needed. Check an install with `python src/selftest.py`. **[RUNBOOK.md](RUNBOOK.md)** is the full step-by-step guide.

## How it works

| Part | File | What it does |
|---|---|---|
| Colour + level measurement | `src/vialcolor.py` | Beer-Lambert absorbance against an empty-vial reference, on linearised sRGB, so the backdrop, glass and lighting cancel. Fill level from the air/liquid absorbance step. |
| Per-vial events | `src/fillevents.py` | Turns the frame stream into one event per finished vial; keeps the colour of the **first liquid to land**, the most direct evidence of carryover. |
| Expected colour + carryover | `src/colormodel.py`, `src/carryover.py` | Tank dye coefficients fitted on clean (last-of-batch) vials; the shared mixing path simulated as a well-mixed hold-up volume (fitted: 48 mL) fed in the fixed T-200 → T-400 → T-100 order. Explains a vial if modelling the previous recipe's residue removes most of its error. |
| Changeover risk | `ColorModel.transition_risk` | Simulates the next recipe after the current one and predicts how far off its first vials will be and how many to flush. |
| AI vial locator | `src/yolo_locator.py`, `models/vial_yolo.onnx` | YOLO11n trained **only on digital-twin renders**; finds the vial under the nozzle for Auto-align. |
| Dashboard | `src/dashboard.py`, `src/dashboard.html` | Standard-library web server + OpenCV; no web framework. |

### Training data from the digital twin

`twin/synth_dataset.py` runs in Isaac Sim on spark7 against the fill-station twin. Every image gets
a random moment of the fill animation, a random liquid colour and opacity (blends of the clear,
orange and blue tanks) and random lighting. Vial and nozzle boxes come from Replicator; the liquid
box is projected from the liquid mesh's bounds through the FillCam calibration (Replicator's tight
boxes skip transparent materials). `twin/get_training_data.py` launches it and copies the set back;
`src/train_yolo.py` trains on it and tests on real frames.

## Results

**Carryover detection** (leave-one-recipe-out on the 30 labelled vials of video 5, `src/calibrate_model.py`):
recall on defective vials **8/9**, cause explanation correct **8/8**; false alarms on clean vials
**9/18**. The false alarms are mostly saturated blue recipes whose real colour is brighter than the
linear dye model predicts. They are reported separately, as the event guide asks.

**AI vial locator** (252 twin renders for training, tested on 300 real frames it never saw):
vial found **98%**, centre error **1.0 px** median. Auto-align (YOLO coarse + template refine)
lands within **1 px** on all 98 test frames with simulated camera shifts of up to 90 px; template
matching alone loses the vial beyond ~60 px. The detector's *liquid* class does **not** transfer to
real frames (55% accuracy; the twin's liquid looks different), so it is not used.

## Reproduce from the dataset

The dataset is not committed. Download it from the event's Fluid Mixing Station folder into `data/`
(`data/vial_fill/{images,labels,videos}`), then:

```bash
python src/build_reference.py         # empty-vial reference from video 5
python src/calibrate_model.py --refresh   # detect vials, fit the model, leave-one-recipe-out eval
python src/dashboard.py --source data/vial_fill/videos/vial_fill_video_5.mp4 --video5 --speed 8
python -m pip install -r requirements-train.txt
python src/train_yolo.py              # after rendering twin data into data/twin_yolo/
```

## Credits

- Data and digital twin: Rockwell Automation, MSOE AI Club and UWM Connected Systems Institute (Fluid Mixing Station dataset, CSI digital twin v1).
- NVIDIA Isaac Sim / Omniverse Replicator for synthetic data; spark7 DGX Spark for rendering.
- Ultralytics YOLO11 (AGPL-3.0); OpenCV; NumPy.
- Built with help from Claude Code.

Event guide: <https://msoe-maic.com/library/?article=Rockwell-NVIDIA-Fall-2026>
