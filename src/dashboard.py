"""Operator dashboard: live vial colour vs recipe, carryover explanations, changeover risk.

    python src/dashboard.py                                   # then open http://127.0.0.1:8050
    python src/dashboard.py --source data/vial_fill/videos/vial_fill_video_5.mp4 --video5
    python src/dashboard.py --source 0                        # USB camera 0
    python src/dashboard.py --source rtsp://host/stream       # network camera
    python src/dashboard.py --source D:/line_capture          # folder; new images are picked up
    python src/dashboard.py --list-cameras                    # which camera numbers exist here

Every source goes through the same path: frame -> vialcolor.measure -> FillTracker ->
one event per finished vial -> ColorModel.judge against the recipe queue. The queue is
what the operator says will run, in order; the n-th finished vial is assigned to the
n-th slot. Verdicts are recomputed from the stored events whenever the queue changes,
so a recipe typed in late still gets judged correctly.

On a new camera or new lighting: park an EMPTY vial under the nozzle, press
"Auto-align" (shifts the frame so the vial lands on the measured geometry) and then
"Capture empty reference" (replaces the Beer-Lambert I0 with one taken on site).
"""
from __future__ import annotations

import argparse, base64, csv, json, os, re, sys, threading, time, webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vialcolor
from vialcolor import measure, VIAL_X, VIAL_Y, CORE_X, COLLAR
from fillevents import FillTracker
from colormodel import ColorModel, Recipe, hexcolor, DEFAULT_MODEL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VF = os.path.join(ROOT, "data", "vial_fill")
SESSIONS = os.path.join(ROOT, "out", "line_test")
SITE_REF = os.path.join(ROOT, "out", "empty_vial_reference_site.npy")
SITE_CAL = os.path.join(ROOT, "out", "site_calibration.json")   # camera offset, kept across restarts
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
STAMP = re.compile(r"(\d{8})_(\d{6})_(\d{6})")   # vial_fill_20260923_195853_579357.jpg
CAMERA_HZ = 6.0                                  # measurement rate on a live camera
ALIGN_SEARCH = 60                                # px either way the auto-align looks
YOLO_EVERY = 4                                   # run the AI locator on every Nth measured frame


# ------------------------------------------------------------------ frame sources
def _stamp(path: str) -> float:
    m = STAMP.search(os.path.basename(path))
    if m:
        return datetime.strptime("".join(m.groups()), "%Y%m%d%H%M%S%f").timestamp()
    return os.path.getmtime(path)


def frames(source: str, every: int, speed: float, stop: threading.Event):
    """Yield (t_seconds, BGR frame, label) from a camera, stream, video file or folder."""
    if os.path.isdir(source):
        seen, t0 = set(), None
        while not stop.is_set():
            new = sorted((p for p in (os.path.join(source, n) for n in os.listdir(source))
                          if p.lower().endswith(IMG_EXT) and p not in seen), key=_stamp)
            for p in new:
                if stop.is_set():
                    return
                img = cv2.imread(p)
                seen.add(p)
                if img is None:
                    continue
                t = _stamp(p)
                t0 = t if t0 is None else t0
                yield t - t0, img, os.path.basename(p)
                if speed > 0:
                    time.sleep(min(2.0, 0.5 / speed))
            time.sleep(0.5)                       # watch for images still being written
        return

    live = source.isdigit() or "://" in source
    cap = cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source {source!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i, t_start, last = 0, time.monotonic(), -1e9
    try:
        while not stop.is_set():
            if live:
                ok, img = cap.read()
                if not ok:
                    time.sleep(0.05); continue
                t = time.monotonic() - t_start
                if t - last < 1.0 / CAMERA_HZ:
                    continue
                last = t
                yield t, img, f"live {t:.1f}s"
            else:
                if not cap.grab():
                    return
                if i % every == 0:
                    ok, img = cap.retrieve()
                    if ok:
                        t = i / fps
                        if speed > 0:                 # play back no faster than speed x real time
                            lag = t / speed - (time.monotonic() - t_start)
                            if lag > 0:
                                time.sleep(lag)
                        yield t, img, f"frame {i}"
                i += 1
    finally:
        cap.release()


def to_station_frame(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Resize to the 640x480 the geometry was measured on, then undo the camera offset."""
    if img.shape[:2] != (480, 640):
        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
    if dx or dy:
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        img = cv2.warpAffine(img, M, (640, 480), borderMode=cv2.BORDER_REPLICATE)
    return img


def vial_crop(img: np.ndarray) -> np.ndarray:
    return img[VIAL_Y[0] - 40:VIAL_Y[1] + 10, VIAL_X[0] - 30:VIAL_X[1] + 30].copy()


def jpeg_b64(img: np.ndarray) -> str:
    return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()


def auto_align(img: np.ndarray, template: np.ndarray) -> tuple[int, int, float]:
    """Find where the holder collar + vial sit in `img` relative to the reference geometry."""
    y0, y1 = COLLAR[0] - 20, VIAL_Y[1]
    x0, x1 = COLLAR[2] - 20, COLLAR[3] + 20
    tpl = cv2.cvtColor(template[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    if img.shape[:2] != (480, 640):
        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sy0, sx0 = max(0, y0 - ALIGN_SEARCH), max(0, x0 - ALIGN_SEARCH)
    win = gray[sy0:min(480, y1 + ALIGN_SEARCH), sx0:min(640, x1 + ALIGN_SEARCH)]
    res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, (bx, by) = cv2.minMaxLoc(res)
    return int(sx0 + bx - x0), int(sy0 + by - y0), float(score)


# ------------------------------------------------------------------ app state
class Monitor:
    def __init__(self, model: ColorModel, locator=None):
        self.model = model
        self.locator = locator          # yolo_locator.Locator, or None when not trained
        self.det = None
        self.lock = threading.Lock()
        self.queue: list[dict] = []         # [{"recipe": {...}, "count": 3}]
        self.events: list[dict] = []        # finished vials, raw measurements only
        self.truth: dict[int, str] = {}     # vial index -> "good" | "bad" from the operator
        self.dx = self.dy = 0
        self.every, self.speed = 8, 1.0
        self.source = ""
        self.status = "idle"
        self.error = ""
        self.frame_jpg = None
        self.last_frame = None
        self.last_raw = None
        self.live = {}
        self.tracker = FillTracker()
        self._stop = threading.Event()
        self._thread = None
        self.session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ref_name = "dataset (video 5)"
        if os.path.exists(SITE_REF):
            vialcolor._REF = np.load(SITE_REF).astype(np.float32)
            self.ref_name = "site capture"
        # The site reference is captured in the aligned frame, so it is only valid with the
        # offset it was captured at: restore both together.
        if os.path.exists(SITE_CAL):
            cal = json.load(open(SITE_CAL))
            self.dx, self.dy = int(cal.get("dx", 0)), int(cal.get("dy", 0))

    def save_calibration(self):
        os.makedirs(os.path.dirname(SITE_CAL), exist_ok=True)
        with open(SITE_CAL, "w") as f:
            json.dump(dict(dx=self.dx, dy=self.dy, saved=datetime.now().isoformat(timespec="seconds")), f)

    # ---- source control
    def start(self, source: str):
        self.stop()
        with self.lock:
            self.source, self.error, self.status = source, "", "starting"
            self.tracker = FillTracker()
            self.events, self.truth = [], {}
            self.session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(source, self._stop), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._thread = None
        with self.lock:
            if self.status not in ("error",):
                self.status = "stopped" if self.source else "idle"

    def _run(self, source: str, stop: threading.Event):
        try:
            with self.lock:
                self.status = "running"
            for t, raw, label in frames(source, self.every, self.speed, stop):
                img = to_station_frame(raw, self.dx, self.dy)
                with self.lock:
                    self.last_raw = raw               # auto-align searches the unshifted view
                m = measure(img)
                ev = self.tracker.push(t, m, vial_crop(img))
                n_seen = getattr(self, "_n_seen", 0) + 1
                self._n_seen = n_seen
                if self.locator and n_seen % YOLO_EVERY == 1:
                    self.det = self.locator.detect(img)
                self._show(img, m, label)
                if ev:
                    self._add(ev)
            ev = self.tracker.flush()
            if ev:
                self._add(ev)
            with self.lock:
                self.status = "finished" if not stop.is_set() else "stopped"
        except Exception as exc:                      # surface it on the page, keep serving
            with self.lock:
                self.status, self.error = "error", f"{type(exc).__name__}: {exc}"

    def _show(self, img, m, label):
        vis = img.copy()
        cv2.rectangle(vis, (CORE_X[0], VIAL_Y[0]), (CORE_X[1], VIAL_Y[1]), (0, 220, 255), 1)
        cy0, cy1, cx0, cx1 = COLLAR
        cv2.rectangle(vis, (cx0, cy0), (cx1, cy1), (255, 160, 0), 1)
        if m.liquid:
            cv2.line(vis, (VIAL_X[0] - 8, m.liquid_top_y), (VIAL_X[1] + 8, m.liquid_top_y), (0, 255, 0), 2)
        det = self.det
        if det is not None:
            det.draw(vis)
        txt = (f"{label}  " + ("no vial" if not m.present else
               f"{m.level_ml:.0f} mL  OD {m.od:.2f}" if m.liquid else "empty vial"))
        for col, th in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(vis, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, th)
        with self.lock:
            self.last_frame = img
            self.frame_jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            self.live = dict(label=label, present=m.present, liquid=m.liquid,
                             level_ml=m.level_ml, od=m.od, hex="#%02x%02x%02x" % m.rgb,
                             filling=self.tracker.filling, vials=self.tracker.count,
                             ai=None if det is None else dict(
                                 vial=round(det.vial[4], 2) if det.vial else None,
                                 liquid=round(det.liquid[4], 2) if det.liquid else None))

    def _add(self, ev):
        rec = dict(index=ev.index, start_s=round(ev.start_s, 1), end_s=round(ev.end_s, 1),
                   final_ml=ev.final_ml, absorbance=list(ev.absorbance),
                   first_absorbance=list(ev.first_absorbance), od=ev.od,
                   time=datetime.now().strftime("%H:%M:%S"),
                   image=jpeg_b64(ev.best_frame) if ev.best_frame is not None else None)
        with self.lock:
            self.events.append(rec)
        self.save()

    # ---- recipe queue -> per-vial assignment and verdicts
    def slots(self) -> list[Recipe]:
        out = []
        for q in self.queue:
            out += [Recipe.from_dict(q["recipe"])] * max(1, int(q.get("count", 3)))
        return out

    def judged(self) -> list[dict]:
        # Re-simulating the whole history is O(vials^2); only redo it when inputs change.
        key = (len(self.events), json.dumps(self.queue, sort_keys=True), tuple(sorted(self.truth.items())))
        if getattr(self, "_judged_key", None) == key:
            return self._judged
        self._judged_key, self._judged = key, self._judge()
        return self._judged

    def _judge(self) -> list[dict]:
        slots = self.slots()
        rows = []
        for i, e in enumerate(self.events):
            row = {k: v for k, v in e.items()}
            row["observed_hex"] = hexcolor(e["absorbance"])
            row["first_hex"] = hexcolor(e["first_absorbance"])
            row["truth"] = self.truth.get(e["index"])
            if i >= len(slots):
                row.update(status="NO RECIPE", reasons=["add this vial's recipe to the queue to judge it"])
            else:
                r = slots[i]
                pos = 1                               # 1-based position within its batch
                for s in reversed(slots[:i]):
                    if s.name != r.name:
                        break
                    pos += 1
                row["recipe"] = r.name
                row["recipe_desc"] = r.describe()
                row["pos"] = pos
                row.update(self.model.judge(e["absorbance"], r, slots[:i],
                                            first_obs=e["first_absorbance"], pos_in_batch=pos))
            rows.append(row)
        return rows

    def risks(self) -> list[dict]:
        out = []
        for a, b in zip(self.queue, self.queue[1:]):
            ra, rb = Recipe.from_dict(a["recipe"]), Recipe.from_dict(b["recipe"])
            if ra.name == rb.name or ra.total <= 0 or rb.total <= 0:
                continue
            out.append(self.model.transition_risk(ra, rb, int(a.get("count", 3))))
        return out

    def accuracy(self, rows) -> dict:
        c = dict(tp=0, fp=0, fn=0, tn=0)
        for r in rows:
            if r.get("truth") not in ("good", "bad") or r["status"] == "NO RECIPE":
                continue
            flagged = r["status"] != "OK"
            bad = r["truth"] == "bad"
            c[("tp" if flagged else "fn") if bad else ("fp" if flagged else "tn")] += 1
        n = sum(c.values())
        c["n"] = n
        c["accuracy"] = round((c["tp"] + c["tn"]) / n, 3) if n else None
        return c

    def state(self) -> dict:
        with self.lock:
            rows = self.judged()
            return dict(status=self.status, error=self.error, source=self.source,
                        dx=self.dx, dy=self.dy, every=self.every, speed=self.speed,
                        reference=self.ref_name, live=self.live,
                        queue=[{**q, "expected_hex": hexcolor(self.model.clean(Recipe.from_dict(q["recipe"])))
                                if Recipe.from_dict(q["recipe"]).total > 0 else None}
                               for q in self.queue],
                        problems={i: Recipe.from_dict(q["recipe"]).problems()
                                  for i, q in enumerate(self.queue)},
                        risks=self.risks(), vials=rows, accuracy=self.accuracy(rows),
                        session=self.session,
                        ai=("YOLO vial locator (trained on digital twin)" if self.locator
                            else "not loaded - run src/train_yolo.py"),
                        model=dict(hold_up_ml=self.model.hold_up, tolerance=self.model.tol,
                                   hydraulics=self.model.p.get("hydraulics")))

    def save(self):
        """Write the session log: one CSV row per vial plus the vial crops."""
        with self.lock:
            rows = self.judged()
        d = os.path.join(SESSIONS, self.session)
        os.makedirs(d, exist_ok=True)
        fields = ["index", "time", "recipe", "pos", "status", "observed_hex", "expected_hex",
                  "predicted_hex", "first_hex", "error_clean", "error_carryover", "final_ml",
                  "truth", "reasons"]
        with open(os.path.join(d, "vials.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({**r, "reasons": " | ".join(r.get("reasons", []))})
        for r in rows:
            p = os.path.join(d, f"vial_{r['index']:03d}.jpg")
            if r.get("image") and not os.path.exists(p):
                with open(p, "wb") as f:
                    f.write(base64.b64decode(r["image"]))
        with open(os.path.join(d, "session.json"), "w") as f:
            json.dump(dict(source=self.source, queue=self.queue, dx=self.dx, dy=self.dy,
                           reference=self.ref_name, truth=self.truth), f, indent=2)

    # ---- site calibration
    def capture_reference(self) -> str:
        with self.lock:
            img = self.last_frame
        if img is None:
            return "no frame yet - start a source first"
        m = measure(img)
        if not m.present:
            return "no vial under the nozzle - park an EMPTY vial and try again"
        if m.liquid and m.od >= 0.2:
            return f"vial does not look empty (OD {m.od:.2f}) - use an empty vial"
        vialcolor._REF = img.astype(np.float32)
        os.makedirs(os.path.dirname(SITE_REF), exist_ok=True)
        np.save(SITE_REF, img)
        cv2.imwrite(SITE_REF.replace(".npy", ".png"), img)
        with self.lock:
            self.ref_name = "site capture"
        self.save_calibration()
        return f"empty-vial reference captured from the current frame (offset dx={self.dx} dy={self.dy} saved with it)"

    def reset_reference(self) -> str:
        for p in (SITE_REF, SITE_CAL):
            if os.path.exists(p):
                os.remove(p)
        vialcolor._REF = None
        with self.lock:
            self.dx = self.dy = 0
        with self.lock:
            self.ref_name = "dataset (video 5)"
        return "back to the dataset reference and zero camera offset"

    def align(self) -> str:
        with self.lock:
            raw = self.last_raw
        if raw is None:
            return "no frame yet - start a source first"
        tpl = np.load(vialcolor._REF_PATH).astype(np.uint8)
        # YOLO finds the vial anywhere in the frame; template matching then pins it to the
        # pixel. Together they recover shifts template matching alone cannot reach.
        cx = self.locator.offset_x(raw) if self.locator else None
        pre = to_station_frame(raw, cx, 0) if cx else raw
        dx, dy, score = auto_align(pre, tpl)
        dx += cx or 0
        if score < 0.4:
            return f"could not find the vial holder (match {score:.2f}); set dx/dy by hand"
        with self.lock:
            self.dx, self.dy = dx, dy
        self.save_calibration()
        how = "YOLO + template" if cx is not None else "template"
        return f"aligned: dx={dx} dy={dy} ({how}, match {score:.2f})"


def list_cameras(n: int = 6):
    """Open camera indices 0..n-1 and save one frame from each, so the vial camera can be picked."""
    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    found = []
    for i in range(n):
        cap = cv2.VideoCapture(i)
        ok, img = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if ok and img is not None:
            p = os.path.join(ROOT, "out", f"camera_{i}.jpg")
            cv2.imwrite(p, img)
            found.append(i)
            print(f"camera {i}: {img.shape[1]}x{img.shape[0]}  snapshot -> {os.path.relpath(p, ROOT)}")
    if not found:
        print("no cameras found on indices 0-%d" % (n - 1))
    else:
        print("\nopen the snapshots, pick the one looking at the fill nozzle, then:\n"
              "  python src/dashboard.py --source <number>")


def recipe_library() -> list[dict]:
    """The labelled recipes from video 5, for one-click entry."""
    p = os.path.join(VF, "labels", "video5_vial_recipes.csv")
    if not os.path.exists(p):
        return []
    lib = {}
    for r in csv.DictReader(open(p)):
        lib.setdefault(r["recipe_id"], dict(name=f"Recipe {r['recipe_id']}",
                                            t100=float(r["t100_ml"] or 0), t200=float(r["t200_ml"] or 0),
                                            t400=float(r["t400_ml"] or 0), note=r["color_purpose"]))
    return list(lib.values())


def video5_queue() -> list[dict]:
    p = os.path.join(VF, "labels", "video5_vial_recipes.csv")
    if not os.path.exists(p):
        return []
    q = []
    for r in csv.DictReader(open(p)):
        name = f"Recipe {r['recipe_id']}"
        if q and q[-1]["recipe"]["name"] == name:
            q[-1]["count"] += 1
        else:
            q.append(dict(recipe=dict(name=name, t100=float(r["t100_ml"] or 0),
                                      t200=float(r["t200_ml"] or 0), t400=float(r["t400_ml"] or 0)),
                          count=1))
    return q


# ------------------------------------------------------------------ web server
class Server(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second copy bind the same port and silently share it;
    # refuse instead, so "port in use" is an error and not two dashboards answering at random.
    allow_reuse_address = os.name != "nt"
    daemon_threads = True


def make_handler(mon: Monitor):
    page = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html"),
                encoding="utf-8").read()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            b = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                return self._send(200, page, "text/html; charset=utf-8")
            if path == "/api/state":
                return self._send(200, json.dumps(mon.state()))
            if path == "/api/library":
                return self._send(200, json.dumps(recipe_library()))
            if path == "/frame.jpg":
                with mon.lock:
                    jpg = mon.frame_jpg
                if jpg is None:
                    return self._send(204, b"")
                return self._send(200, jpg, "image/jpeg")
            self._send(404, json.dumps({"error": "not found"}))

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._send(400, json.dumps({"error": "bad json"}))
            path = urlparse(self.path).path
            msg = "ok"
            if path == "/api/start":
                with mon.lock:
                    mon.every = int(body.get("every", mon.every))
                    mon.speed = float(body.get("speed", mon.speed))
                mon.start(str(body.get("source", "")).strip().strip('"'))
            elif path == "/api/stop":
                mon.stop()
            elif path == "/api/queue":
                with mon.lock:
                    mon.queue = body.get("queue", [])
                mon.save()
            elif path == "/api/video5":
                with mon.lock:
                    mon.queue = video5_queue()
            elif path == "/api/shift":
                with mon.lock:
                    mon.dx, mon.dy = int(body.get("dx", 0)), int(body.get("dy", 0))
                mon.save_calibration()
            elif path == "/api/align":
                msg = mon.align()
            elif path == "/api/reference":
                msg = mon.reset_reference() if body.get("reset") else mon.capture_reference()
            elif path == "/api/truth":
                with mon.lock:
                    i, v = int(body["index"]), body.get("value")
                    if v in ("good", "bad"):
                        mon.truth[i] = v
                    else:
                        mon.truth.pop(i, None)
                mon.save()
            else:
                return self._send(404, json.dumps({"error": "not found"}))
            self._send(200, json.dumps({"message": msg}))

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="camera index, stream URL, video file or image folder")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to let other laptops on the network view it")
    ap.add_argument("--every", type=int, default=8, help="video files: measure every Nth frame")
    ap.add_argument("--speed", type=float, default=1.0, help="video files: playback speed, 0 = as fast as possible")
    ap.add_argument("--video5", action="store_true", help="pre-load the video-5 recipe schedule")
    ap.add_argument("--no-yolo", action="store_true", help="skip the AI vial locator even if trained")
    ap.add_argument("--no-browser", action="store_true", help="do not open the page automatically")
    ap.add_argument("--list-cameras", action="store_true", help="snapshot every camera and exit")
    a = ap.parse_args()
    if a.list_cameras:
        return list_cameras()

    locator = None
    if not a.no_yolo:
        try:
            from yolo_locator import Locator, available
            if available():
                locator = Locator()
                print(f"AI vial locator loaded ({os.path.relpath(locator.weights, ROOT)}, {locator.backend})")
            else:
                print("AI vial locator not available - Auto-align uses template matching only")
        except Exception as exc:                  # the dashboard must run without it
            print(f"AI vial locator unavailable: {exc}")
    mon = Monitor(ColorModel.load(a.model), locator)
    mon.every, mon.speed = a.every, a.speed
    if a.video5:
        mon.queue = video5_queue()
    if a.source:
        mon.start(a.source)
    try:
        srv = Server((a.host, a.port), make_handler(mon))
    except OSError as exc:
        mon.stop()
        sys.exit(f"cannot listen on port {a.port} ({exc.strerror or exc}) - another dashboard is "
                 f"probably running; close it or use --port {a.port + 10}")
    print(f"dashboard on http://{'127.0.0.1' if a.host == '0.0.0.0' else a.host}:{a.port}   (Ctrl+C to stop)")
    print(f"session logs -> {os.path.relpath(SESSIONS, ROOT)}{os.sep}")
    if not a.no_browser:
        threading.Timer(1.0, webbrowser.open, (f"http://127.0.0.1:{a.port}",)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        mon.stop()
        mon.save()


if __name__ == "__main__":
    main()
