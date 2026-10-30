"""Check that the monitor works on this machine, with no dataset and no camera needed.

    python src/selftest.py              # every check, PASS / FAIL each, exit code 1 on any FAIL
    python src/selftest.py --cameras    # also open cameras 0-5 and save a snapshot of each

Each check exercises the real code path the dashboard uses:
  1. Python and packages            5. carryover model verdicts and changeover risk
  2. shipped model files            6. AI vial locator (YOLO) loads and finds the vial
  3. empty vial reads as empty      7. Auto-align recovers a moved camera
  4. a synthetic dyed fill is measured with the right level and colour
  8. the dashboard web server starts and answers
"""
from __future__ import annotations

import argparse, json, os, sys, threading, traceback, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
RESULTS = []


def check(name):
    def wrap(fn):
        def run():
            try:
                detail = fn()
                RESULTS.append((name, "PASS", detail or ""))
            except Exception as exc:                  # every check reports, none stops the rest
                RESULTS.append((name, "FAIL", f"{type(exc).__name__}: {exc}"))
                if os.environ.get("SELFTEST_TRACE"):
                    traceback.print_exc()
        run.__name__ = fn.__name__
        return run
    return wrap


@check("1 python + packages")
def t_packages():
    assert sys.version_info >= (3, 10), f"Python {sys.version.split()[0]} - need 3.10 or newer"
    import cv2, numpy
    return f"Python {sys.version.split()[0]}, OpenCV {cv2.__version__}, NumPy {numpy.__version__}"


@check("2 model files")
def t_files():
    need = ["models/color_model.json", "models/empty_vial_reference.npy", "src/dashboard.html"]
    missing = [p for p in need if not os.path.exists(os.path.join(ROOT, p))]
    assert not missing, f"missing {missing} - is this a complete clone?"
    opt = [p for p in ("models/vial_yolo.onnx", "models/vial_yolo.pt") if os.path.exists(os.path.join(ROOT, p))]
    return "all present; AI weights: " + (", ".join(opt) if opt else "NONE (Auto-align falls back to template)")


def _reference():
    import numpy as np, vialcolor
    return np.load(vialcolor._REF_PATH).astype(np.uint8)


@check("3 empty vial reads empty")
def t_empty():
    from vialcolor import measure
    m = measure(_reference())
    assert m.present, "no vial detected in the reference frame"
    assert not (m.liquid and m.od >= 0.2), f"reference reads as liquid (OD {m.od})"
    return "vial present, no liquid"


def synthetic_fill(top_y: int, absorbance_rgb):
    """The reference frame with a dye of known absorbance filled up to top_y (Beer-Lambert)."""
    import numpy as np
    from vialcolor import srgb_to_linear, linear_to_srgb, VIAL_X, VIAL_Y
    img = _reference().astype(np.float64)
    x0, x1 = VIAL_X
    t = np.power(10.0, -np.asarray(absorbance_rgb, float))[::-1]        # RGB -> BGR order
    region = srgb_to_linear(img[top_y:VIAL_Y[1], x0:x1]) * t
    img[top_y:VIAL_Y[1], x0:x1] = linear_to_srgb(region)
    return np.clip(img, 0, 255).astype(np.uint8)


@check("4 synthetic fill measured")
def t_fill():
    import numpy as np
    from vialcolor import measure, VIAL_Y, VIAL_CAPACITY_ML
    top, A = 320, (0.25, 0.75, 1.10)
    m = measure(synthetic_fill(top, A))
    expect_ml = (VIAL_Y[1] - top) / (VIAL_Y[1] - VIAL_Y[0]) * VIAL_CAPACITY_ML
    assert m.liquid, "liquid not detected"
    assert abs(m.level_ml - expect_ml) < 6, f"level {m.level_ml} mL, expected ~{expect_ml:.0f}"
    err = float(np.max(np.abs(np.array(m.absorbance) - A)))
    assert err < 0.15, f"absorbance {m.absorbance}, expected {A}"
    return f"level {m.level_ml:.0f} mL (expected {expect_ml:.0f}), absorbance error {err:.3f}"


@check("5 carryover model")
def t_model():
    from colormodel import ColorModel, Recipe
    cm = ColorModel.load()
    orange = Recipe("Recipe 6", t100=30, t200=10)
    blue = Recipe("Recipe 7", t100=20, t400=50)
    ok = cm.judge(cm.clean(blue), blue, [blue, blue])
    assert ok["status"] == "OK", f"clean blue vial judged {ok['status']}"
    dirty = cm.with_history([orange] * 3, blue)
    v = cm.judge(dirty, blue, [orange] * 3)
    assert v["status"].startswith("CARRYOVER"), f"orange->blue first vial judged {v['status']}"
    r = cm.transition_risk(orange, blue)
    assert r["level"] in ("MEDIUM", "HIGH"), f"orange->blue risk {r['level']}"
    bad = Recipe("bad", t100=40, t200=60).problems()
    assert bad, "a recipe over the T-200 limit was not flagged"
    return f"clean=OK, after orange={v['status']}, risk {r['level']}, limits enforced"


@check("6 AI vial locator")
def t_yolo():
    from yolo_locator import Locator, available
    if not available():
        return "SKIPPED - no AI weights; dashboard still runs (template Auto-align)"
    loc = Locator()
    assert loc.home is not None, "YOLO did not find the vial in the reference frame"
    d = loc.detect(synthetic_fill(300, (0.3, 0.9, 1.2)))
    assert d.vial is not None, "YOLO did not find the vial in a filled frame"
    return f"{loc.backend} backend, vial found (conf {d.vial[4]:.2f})"


@check("7 auto-align recovers moved camera")
def t_align():
    import numpy as np, cv2
    from dashboard import auto_align, to_station_frame
    from yolo_locator import Locator, available
    img = synthetic_fill(320, (0.25, 0.75, 1.10))
    tpl = _reference()
    loc = Locator() if available() else None
    worst = 0
    for dx, dy in ((40, -10), (-55, 12), (0, 0)):
        moved = cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]), (640, 480),
                               borderMode=cv2.BORDER_REPLICATE)
        cx = loc.offset_x(moved) if loc else None
        pre = to_station_frame(moved, cx, 0) if cx else moved
        fx, fy, score = auto_align(pre, tpl)
        assert score >= 0.4, f"alignment lost the vial at shift ({dx},{dy})"
        worst = max(worst, abs(fx + (cx or 0) - dx), abs(fy - dy))
    assert worst <= 3, f"alignment off by {worst} px"
    return f"worst error {worst} px over shifts up to 55 px" + ("" if loc else " (template only)")


@check("8 dashboard server")
def t_server():
    from http.server import ThreadingHTTPServer
    import dashboard
    from colormodel import ColorModel
    mon = dashboard.Monitor(ColorModel.load())
    srv = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.make_handler(mon))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        page = urllib.request.urlopen(base + "/", timeout=10).read()
        assert b"Recipe-Aware" in page, "page did not render"
        state = json.loads(urllib.request.urlopen(base + "/api/state", timeout=10).read())
        assert "vials" in state and "risks" in state, "state API incomplete"
    finally:
        srv.shutdown()
    return "page and /api/state OK"


def cameras():
    import dashboard
    dashboard.list_cameras()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cameras", action="store_true", help="also snapshot cameras 0-5")
    a = ap.parse_args()
    for t in (t_packages, t_files, t_empty, t_fill, t_model, t_yolo, t_align, t_server):
        t()
    w = max(len(n) for n, _, _ in RESULTS)
    print()
    for name, status, detail in RESULTS:
        print(f"  [{status}] {name:<{w}}  {detail}")
    failed = [n for n, s, _ in RESULTS if s == "FAIL"]
    print(f"\n{'ALL CHECKS PASSED' if not failed else f'{len(failed)} CHECK(S) FAILED: ' + ', '.join(failed)}")
    if failed:
        print("re-run with SELFTEST_TRACE=1 for full tracebacks")
    if a.cameras:
        print("\ncameras:")
        cameras()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
