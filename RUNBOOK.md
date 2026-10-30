# RUNBOOK: running the Recipe-Aware Color Monitor on the fill station

This runbook is written for whoever runs the system on the factory computer: a person, or an AI
coding agent acting for them. Every step says what to run, what you should see, and what to do if
you see something else. Follow it top to bottom. Do not skip **Step 3 (self-test)**.

---

## 0. What this system does, and what it must never do

**Does:** watches the fill-station camera (the `vial_fill` view: the vial under the dispensing
nozzle), measures each finished vial's colour and fill level, compares it with the colour its
recipe should produce, flags mismatches, explains likely causes (carryover from the previous
recipe in the shared mixing path), and predicts risky recipe changeovers before they run. It
shows all of this on a local web page and logs every vial.

**Never:** it is **read-only**. It does not talk to the PLC, the Batch Scheduler or any station
control. It only reads camera images.

Rules for the person or agent operating it:

- **Do not change recipes, tank settings or the queue on the station.** Recipe changes need the
  plant manager's guidance. The recipe list *in the dashboard* is only what we are told will run.
- **Station limits** (the dashboard flags recipes that break them; do not ask anyone to run them):
  a vial holds at most **80 mL**; each used tank at least **10 mL**; **T-200 at most 50 mL**
  (more can fault the system); **never use T-300** (out of service).
- Tanks: **T-100 = clear diluent, T-200 = orange, T-400 = blue**. Dispense order is fixed:
  T-200 → T-400 → T-100. If the tank colours on the day are different, tell the team (see §11).
- Do not install anything system-wide or change system settings. Use the virtual environment in
  Step 2. Ask the operator before closing other programs that might be using the camera.

### 0b. When something goes wrong: what you may and may not change

The goal on the day is an honest test of *this* system. Keeping it running is good; changing what it
concludes is not.

**You may fix (setup and robustness):**

- Installing or upgrading packages inside `.venv`, using another Python, the headless OpenCV build.
- Choosing the camera, stream URL, folder or port; setting the camera offset (dx / dy) by hand.
- Camera access, permissions or firewall problems on this computer (with the operator's OK for
  anything outside the repository).
- **Small code fixes that stop a crash or a hang** without changing any result, e.g. an unusual
  camera frame format, a file that cannot be read, a path with spaces, a timeout on a stream.

**You must not change (it would change the results being tested):**

- How colour or level is measured, how vials are judged, or any threshold, tolerance or fitted
  number: `src/vialcolor.py`, `src/colormodel.py`, `src/carryover.py`, `src/fillevents.py`
  constants, and everything in `models/`.
- Do not retrain or refit anything on site, and do not edit results or operator truth after the fact.
- Anything on the station: recipes, tanks, PLC, Batch Scheduler (see the rules above).

If results look wrong (false alarms, a level that reads high), **record them as they are**. That is
part of what is being measured. Note what you saw in the session notes (below).

**Log every change.** Before the run, and again after any fix, save the state of the code into the
session folder:

```bash
python -c "import os; os.makedirs('out/line_test', exist_ok=True)"
git status > out/line_test/CHANGES.txt
git diff >> out/line_test/CHANGES.txt
```

(If the code came as a ZIP without git, list the changed files and paste the changed lines into
`CHANGES.txt` by hand.)

Then add a short note at the top of `CHANGES.txt`: what broke, the error message, what you changed
and why. Include the file in what you send back (§11).

**Stop rule.** If two attempts do not fix a problem, or the only fix would be one of the
"must not change" items, stop. Report the exact error, what you tried, and a screenshot of the
camera view (§11) instead of working around it.

---

## 1. Requirements

| Need | Detail |
|---|---|
| OS | Windows 10/11, macOS 12+, or Linux |
| Python | **3.10 or newer**. Check with `python --version` (Windows) / `python3 --version` (Mac/Linux). |
| Internet | Only for the install in Step 2 (about 60 MB: NumPy + OpenCV). Running needs none. |
| Camera | The fill-station camera visible to this computer as (a) a camera device, (b) an `rtsp://` / `http://` stream, or (c) a folder that the vision system saves images into. A recorded video file also works. |
| Browser | Any modern browser on this computer. |
| GPU | Not needed. Everything runs on the CPU. |

No dataset download is needed. The trained model, the empty-vial reference and the AI vial
locator are all in the repository under `models/`.

---

## 2. Get the code and install

```bash
git clone https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-michael-wood-s-team-1-2ccd0bfbe8.git color-monitor
cd color-monitor
```

If git is not available, download the repository as a ZIP from GitHub (green **Code** button →
**Download ZIP**), unzip it, and `cd` into the folder. Every command below runs **from the
repository root** (the folder that contains `RUNBOOK.md`).

Create an isolated environment and install:

**Windows (PowerShell or cmd):**
```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Expected: pip ends with `Successfully installed numpy-... opencv-python-...` (or says they are
already satisfied). From here on, `python` means the one inside `.venv`. Re-activate it in any new
terminal.

| If you see | Do this |
|---|---|
| `python` not found / opens the Microsoft Store | Use `py -3` instead of `python` on Windows, or install Python 3.10+ from python.org |
| `No matching distribution found for opencv-python` | Python is too old or 32-bit; use 64-bit Python 3.10+ |
| SSL / proxy errors | The network blocks PyPI. Try a phone hotspot for the install only, or `pip download -r requirements.txt` on another machine and `pip install --no-index --find-links <folder> -r requirements.txt` here |
| Linux: `ImportError: libGL.so.1` | `pip uninstall opencv-python && pip install opencv-python-headless` (the dashboard does not need OpenCV's own windows) |

---

## 3. Self-test (must pass before touching the line)

```bash
python src/selftest.py
```

Takes under a minute. Expected final output:

```
  [PASS] 1 python + packages                 Python 3.x, OpenCV 4.x/5.x, NumPy ...
  [PASS] 2 model files                       all present; AI weights: models/vial_yolo.onnx, ...
  [PASS] 3 empty vial reads empty            vial present, no liquid
  [PASS] 4 synthetic fill measured           level 52 mL (expected 52), absorbance error 0.0xx
  [PASS] 5 carryover model                   clean=OK, after orange=CARRYOVER, risk HIGH, limits enforced
  [PASS] 6 AI vial locator                   onnx backend, vial found (conf 0.8x)
  [PASS] 7 auto-align recovers moved camera  worst error 0 px over shifts up to 55 px
  [PASS] 8 dashboard server                  page and /api/state OK

ALL CHECKS PASSED
```

The command exits with code 0 on success and 1 on any failure. For full error details:
`SELFTEST_TRACE=1 python src/selftest.py` (PowerShell: `$env:SELFTEST_TRACE=1; python src/selftest.py`).

| Failing check | Meaning | Fix |
|---|---|---|
| 1 | Wrong Python or packages missing | Back to Step 2; make sure `.venv` is activated |
| 2 | Incomplete download | Re-clone. `models/` must contain `color_model.json`, `empty_vial_reference.npy`, `vial_yolo.onnx` |
| 3, 4 | Colour measurement broken | Do not run on the line. Send the output to the team (§11) |
| 5 | Model file corrupt | Re-clone |
| 6 | AI locator cannot load (old OpenCV without ONNX support) | `pip install -U opencv-python`. If it still fails, carry on: the dashboard runs without it and Auto-align uses template matching (good for camera shifts up to about 60 px) |
| 7 | Alignment broken | Usually follows from 6; same fix |
| 8 | Port in use or firewall | Close other copies of the dashboard; or run it later with `--port 8060` |

Check 6 can show `SKIPPED`. That is acceptable: the monitor still works.

---

## 4. Find the fill-station camera

### 4a. Camera plugged into this computer (USB / capture card)

```bash
python src/dashboard.py --list-cameras
```

It opens cameras 0–5 and saves one picture of each as `out/camera_<N>.jpg`, printing e.g.
`camera 1: 640x480  snapshot -> out\camera_1.jpg`. **Open the snapshots** and pick the one that
shows the following, which is the view the system was built on (compare with
`models/empty_vial_reference.png`):

- a clear cylindrical vial standing in a **black holder block** in the centre of the frame;
- a thin **dispensing nozzle** coming down from the top centre, above the vial;
- a **white paper backdrop** in the upper half, dark machine parts and track at the bottom;
- often a second vial on its cart to the right.

Note its number `N`. A laptop's built-in webcam usually shows up as camera 0; that is **not** the
station camera.

If no camera is found: check the cable, close any other program using the camera (vision
software, Teams/Zoom, Camera app), and on macOS allow camera access for the terminal in
*System Settings → Privacy & Security → Camera* (ask the operator to do it).

### 4b. Network camera

Use its stream URL as the source, e.g. `rtsp://<ip>:554/stream1` or `http://<ip>/video.mjpg`.
Ask the operator or plant staff for the URL. Credentials go inside the URL only if the operator
types them in.

### 4c. The vision system saves images to a folder

Use the folder path as the source. The dashboard reads the images already there in time order,
then picks up new ones as they appear (`.jpg .jpeg .png .bmp`). Times are read from file names
like `vial_fill_20260923_195853_579357.jpg`, otherwise from file modification time.

### 4d. Recorded video

Any `.mp4` / `.avi` path works. Use **speed** in the page (1 = real time, 0 = as fast as possible).

**Any resolution is accepted** (frames are resized to 640×480). The view must, however, be framed
like the dataset camera: roughly the same angle and zoom. Small shifts are corrected in Step 6;
a camera at a very different zoom or angle is not supported (see §10).

---

## 5. Start the dashboard

```bash
python src/dashboard.py --source <N or URL or folder or video>
```

or with no `--source` to choose it in the page. On Windows `run_dashboard.bat` does the install
check and asks for the source; on Mac/Linux `./run_dashboard.sh <source>`.

Expected console output:

```
AI vial locator loaded (models\vial_yolo.onnx, onnx)
dashboard on http://127.0.0.1:8050   (Ctrl+C to stop)
session logs -> out\line_test\
```

A browser tab opens at **http://127.0.0.1:8050**. If it does not, open that address by hand.

Useful options:

| Option | Use |
|---|---|
| `--port 8060` | 8050 is taken |
| `--host 0.0.0.0` | let other computers on the network view the page at `http://<this-ip>:8050` |
| `--no-browser` | do not open a browser tab (headless / agent use) |
| `--no-yolo` | disable the AI locator |
| `--speed 0 --every 4` | video files: process as fast as possible, every 4th frame |
| `--video5` | pre-load the dataset's recipe schedule (only useful with the dataset video) |

The page header shows the status (`running`, `stopped`, `error: ...`). The camera panel shows the
live frame with overlays: a **yellow box** (where colour is measured), a **blue box** (holder
block), a **green line** (detected liquid surface) and **AI vial** boxes from YOLO.

---

## 6. Calibrate on site (about 1 minute, do it every time)

Do this with an **empty vial parked under the nozzle**. Ask the operator to park one.

1. **Auto-align.** Press **Auto-align**. The message reads like
   `aligned: dx=12 dy=-4 (YOLO + template, match 0.85)`.
   - Check the **yellow box sits inside the vial glass**, below the black holder block, and the
     blue box is on the holder block. If not, adjust **dx / dy** by hand (positive dx moves the
     measurement right, positive dy moves it down) and press **Apply**.
   - `could not find the vial holder` → the view is too different; see §10.
2. **Capture empty reference.** Press **Capture empty reference**. Expected message:
   `empty-vial reference captured from the current frame (offset dx=.. dy=.. saved with it)`.
   The footer then shows `reference: site capture`. This cancels the local lighting and glass.
   The reference **and** the camera offset are saved (`out/empty_vial_reference_site.npy`,
   `out/site_calibration.json`) and restored on the next start. **Reset ref** deletes both and
   goes back to the dataset reference with zero offset.
   - `vial does not look empty` → the vial has liquid; park an empty one.
   - `no vial under the nozzle` → the cart is not parked or alignment is off.
3. **Sanity check:** with the empty vial, the live readout under the image should say
   **empty** (not a level in mL). When liquid is dispensed it should show a rising level in mL and
   a coloured swatch, with the green line tracking the surface.

Redo both steps if the camera is bumped or the lighting changes.

---

## 7. Enter the recipes that will run

In **Recipe queue**, add one row per recipe **in the order the station will run them**:

- **name**: anything (e.g. `Recipe 7`); a change of name marks a changeover.
- **T-100 / T-200 / T-400**: millilitres from each tank, as set in the Batch Scheduler.
- **vials**: how many vials of that recipe will be filled in a row (default 3).

The **n-th finished vial is judged against the n-th slot**, so the queue must match what actually
runs. If a vial is missed or an extra one is filled, edit the counts: verdicts are recomputed
for every vial immediately.

Rows that break the station limits turn red with the reason. The **expected** swatch is the colour
the recipe should produce. The **Upcoming changeover risk** table (right) appears once two
different recipes are queued:

| Column | Meaning |
|---|---|
| risk | LOW / MEDIUM / HIGH, from how far the first vial is predicted to be from the clean colour |
| should be / 1st vial predicted | clean colour vs colour predicted with residue from the previous recipe |
| vials affected | how many vials of the new recipe are expected off-colour |
| advice | e.g. flush ~48 mL to waste before the changeover |

This risk table is the **prediction before the vial is filled**. Note it before each changeover,
then compare it with what happens.

---

## 8. Run and read the results

Press **Start** (or start with `--source`). Each time a vial finishes filling and the cart moves
on, a row appears in **Inspected vials** (newest first):

| Column | Meaning |
|---|---|
| vial | photo of the vial at the end of its fill |
| recipe | the recipe slot it was judged against, its tank volumes, its position in the batch, the measured level |
| detected | measured colour; *first drop* is the colour of the first liquid that landed (residue shows here) |
| expected | colour the recipe should produce |
| w/ carryover | colour predicted once the previous recipe's residue is included |
| verdict | see below, plus the error numbers *clean → with carryover* |

| Verdict | Meaning |
|---|---|
| **OK** | colour matches the recipe |
| **CARRYOVER** | off-colour, and modelling the previous recipe's residue explains it |
| **CARRYOVER?** | off-colour on the first vial after a changeover, likely residue, but the model does not fully reproduce it |
| **MISMATCH** | off-colour and **not** explained by the previous recipe: check tank contents, dosing, lighting, or the queue order |
| **NO RECIPE** | more vials finished than the queue has slots: add the recipe |

---

## 9. Score it and save the results

For every vial, look at the real vial (or ask the operator) and press **good** or **off-colour**.
The counters at the top show agreement, **caught**, **missed**, **false alarms** and
**correct pass**. Report missed defects and false alarms separately, as the event asks.

Everything is saved continuously to `out/line_test/<session time>/`:

- `vials.csv`: one row per vial (recipe, verdict, colours, errors, operator truth, reasons)
- `vial_NNN.jpg`: the photo of each vial
- `session.json`: source, queue, camera offset, reference used, operator truth

**Starting a new source starts a new session folder.** To stop, press **Stop** in the page and
then `Ctrl+C` in the terminal (results are saved on exit). Zip the session folder and send it to
the team (§11).

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Header says `error: RuntimeError: cannot open source '0'` | Wrong camera number, or camera busy | `--list-cameras` again; close other camera apps |
| Live image black / frozen | Camera busy or stream dropped | Stop, start again; check the stream URL in VLC |
| Readout always `no vial` | Measurement box is not on the holder block | Auto-align with an empty vial; adjust dx/dy |
| Empty vial reads a level in mL | Reference from different lighting | Capture empty reference (Step 6.2) |
| Levels look right but every vial is MISMATCH | Queue does not match what ran, or tank colours differ from T-100 clear / T-200 orange / T-400 blue | Fix the queue; if tank colours differ, report it (the model is calibrated to those dyes) |
| Vials appear in the wrong rows | A vial was missed or extra | Edit the counts in the queue; verdicts recompute |
| No vial rows ever appear although liquid shows | Fill never "finishes": cart does not leave the frame, or the stream stops mid-fill | Rows appear when the cart moves away; stop the source to flush the last vial |
| `could not find the vial holder` | Camera angle or zoom very different from the dataset view | Ask whether the camera can be positioned like `models/empty_vial_reference.png`; this system expects that framing |
| `AI vial locator not available` | ONNX model could not load | Auto-align still works by template matching; or `pip install -U opencv-python` |
| Page does not load | Server not running | Check the terminal for an error; start it again |
| `cannot listen on port 8050 ... another dashboard is probably running` | A copy is already running | Use the page of the running copy, or close it (Ctrl+C in its terminal), or start with `--port 8060` |
| Readings wrong after restarting the computer and moving the camera | Saved offset/reference are from the old position | Press **Reset ref**, then redo Step 6 |
| Very slow | Slow CPU | Video files: raise `every` (frames skipped) in the page; live cameras are capped at 6 measurements per second |

---

## 11. Reporting back

Send to the team: the zipped `out/line_test/<session>/` folder, `out/line_test/CHANGES.txt`, the output of
`python src/selftest.py`, a photo or screenshot of the camera view, and the recipes that ran
(with tank colours if they were not clear / orange / blue).

---

## 12. Driving it without the browser (for agents)

Everything the page does goes through a small JSON API on the same port. With the dashboard
running (`--no-browser` is fine):

```bash
# start a source (camera number, URL, folder or video); speed/every apply to video files
curl -s -X POST localhost:8050/api/start -H "Content-Type: application/json" -d '{"source":"1"}'

# park an EMPTY vial, then align and capture the reference
curl -s -X POST localhost:8050/api/align -H "Content-Type: application/json" -d '{}'
curl -s -X POST localhost:8050/api/reference -H "Content-Type: application/json" -d '{}'

# set the recipe queue (replaces it)
curl -s -X POST localhost:8050/api/queue -H "Content-Type: application/json" -d '{"queue":[
  {"recipe":{"name":"Recipe 6","t100":30,"t200":10,"t400":0},"count":3},
  {"recipe":{"name":"Recipe 7","t100":20,"t200":0,"t400":50},"count":3}]}'

# everything: status, live reading, changeover risks, judged vials, accuracy
curl -s localhost:8050/api/state

# operator truth for vial 4 ("good", "bad", or "" to clear)
curl -s -X POST localhost:8050/api/truth -H "Content-Type: application/json" -d '{"index":4,"value":"bad"}'

# manual camera offset, and stop
curl -s -X POST localhost:8050/api/shift -H "Content-Type: application/json" -d '{"dx":10,"dy":-5}'
curl -s -X POST localhost:8050/api/stop -H "Content-Type: application/json" -d '{}'
```

`GET /frame.jpg` returns the current annotated camera frame. In `/api/state`, `vials[*]` carries
`status`, `reasons`, `observed_hex`, `expected_hex`, `predicted_hex`, `error_clean`,
`error_carryover`, `recipe`, `pos` and `final_ml`; `risks[*]` carries the changeover predictions;
`live` is the current reading; `error` is non-empty if the source failed. On Windows PowerShell,
use `curl.exe` (not `curl`) or `Invoke-RestMethod`.

---

## 13. Known limits (state these honestly when reporting)

- Built and validated on the dataset's `vial_fill` camera view. Shifts are corrected
  automatically; different zoom or angle is not.
- Colour model calibrated for T-100 clear, T-200 orange, T-400 blue. Other dyes need refitting
  (`src/calibrate_model.py` with labelled vials).
- On the dataset (leave-one-recipe-out) it caught 8 of 9 off-colour vials and named the cause
  correctly for 8 of 8, but also flagged 9 of 18 clean vials, mostly saturated blues that read
  brighter than the model predicts. Expect false alarms on strong blue recipes.
- The fill level scale is approximate near the top of the vial (70 mL fills can read ~78–80 mL).
- The AI vial locator was trained only on digital-twin renders; on 300 real frames it found the
  vial 98% of the time with 1 px median centre error. Its liquid detection is not used.
