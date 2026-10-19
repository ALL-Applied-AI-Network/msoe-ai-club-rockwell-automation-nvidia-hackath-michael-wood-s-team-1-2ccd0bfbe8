"""Find the fills in a video from the images alone, with no reliance on the labels.

Writes out/measurements_video<N>.csv (one row per frame) and out/fills_video<N>.csv
(one row per detected vial). Because it never reads the label file it works on the
unlabelled videos 1-4 as well.
"""
import argparse, csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vialcolor import measure

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VF = os.path.join(ROOT, "data", "vial_fill")
OUT = os.path.join(ROOT, "out")

DROP_ML = 12.0        # a fall this size means the cart indexed and a new vial arrived
MIN_FILL_ML = 8.0     # ignore blips that never become a real fill
MIN_FRAMES = 3
MIN_OD = 0.18         # below this the "fill" is glass/noise, not dyed liquid


def run(video: str):
    os.makedirs(OUT, exist_ok=True)
    idx = sorted((r for r in csv.DictReader(open(os.path.join(VF, "image_index.csv")))
                  if r["video_id"] == video), key=lambda r: r["timestamp"])
    print(f"video {video}: {len(idx)} frames")

    rows = []
    t0 = None
    for r in idx:
        p = os.path.join(VF, "images", r["filename"])
        if not os.path.exists(p):
            continue
        m = measure(p)
        ts = r["timestamp"]
        t0 = t0 or ts
        rows.append(dict(filename=r["filename"], timestamp=ts,
                         present=int(m.present), liquid=int(m.liquid),
                         level_ml=m.level_ml, od=m.od, hue_deg=m.hue_deg,
                         chroma=m.chroma, r=m.rgb[0], g=m.rgb[1], b=m.rgb[2],
                         a_r=m.absorbance[0], a_g=m.absorbance[1], a_b=m.absorbance[2],
                         confidence=m.confidence))

    mpath = os.path.join(OUT, f"measurements_video{video}.csv")
    with open(mpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # Segment: a new vial starts when the level drops or liquid disappears.
    fills, cur = [], []
    prev = 0.0
    for r in rows:
        lvl = r["level_ml"] if r["liquid"] else 0.0
        if (prev - lvl) > DROP_ML or (prev > 0 and lvl == 0.0):
            if cur: fills.append(cur)
            cur = []
        if lvl > 0:
            cur.append(r)
        prev = lvl
    if cur: fills.append(cur)

    recs = []
    for i, seg in enumerate(fills, 1):
        peak = max(r["level_ml"] for r in seg)
        if peak < MIN_FILL_ML or len(seg) < MIN_FRAMES:
            continue
        # Final colour = median over the frames within 90% of the peak level.
        top = [r for r in seg if r["level_ml"] >= peak * 0.9]
        peak_frame = max(seg, key=lambda r: r["level_ml"])["filename"]
        od_final = float(np.median([r["od"] for r in top]))
        if od_final < MIN_OD:        # empty glass / no vial, not a dyed fill
            continue
        recs.append(dict(
            fill=len(recs) + 1,
            start=seg[0]["timestamp"], end=seg[-1]["timestamp"], n_frames=len(seg),
            start_ml=round(seg[0]["level_ml"], 1), final_ml=round(peak, 1),
            peak_frame=peak_frame,
            hue_deg=round(float(np.median([r["hue_deg"] for r in top])), 1),
            od=round(float(np.median([r["od"] for r in top])), 3),
            chroma=round(float(np.median([r["chroma"] for r in top])), 1),
            r=int(np.median([r["r"] for r in top])), g=int(np.median([r["g"] for r in top])),
            b=int(np.median([r["b"] for r in top])),
            a_r=round(float(np.median([r["a_r"] for r in top])), 4),
            a_g=round(float(np.median([r["a_g"] for r in top])), 4),
            a_b=round(float(np.median([r["a_b"] for r in top])), 4)))

    fpath = os.path.join(OUT, f"fills_video{video}.csv")
    with open(fpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys())); w.writeheader(); w.writerows(recs)
    print(f"  {len(recs)} fills detected -> {os.path.relpath(fpath, ROOT)}")
    return recs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="5")
    a = ap.parse_args()
    recs = run(a.video)
    print(f"\n{'fill':>4} {'start':>23} {'n':>3} {'ml0':>5} {'final':>6} {'hue':>6} {'OD':>6} {'RGB':>16}")
    for r in recs:
        print(f"{r['fill']:>4} {r['start']:>23} {r['n_frames']:>3} {r['start_ml']:>5.0f} "
              f"{r['final_ml']:>6.0f} {r['hue_deg']:>6.1f} {r['od']:>6.3f} "
              f"{str((r['r'],r['g'],r['b'])):>16}")
