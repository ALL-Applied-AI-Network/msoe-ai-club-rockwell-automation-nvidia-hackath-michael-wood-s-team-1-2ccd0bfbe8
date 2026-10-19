"""Compare image-derived fill windows with the label file's fill windows."""
import csv, os
from datetime import datetime
import numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VF=os.path.join(ROOT,"data","vial_fill")
fills=list(csv.DictReader(open(os.path.join(ROOT,"out","fills_video5.csv"))))
frames=list(csv.DictReader(open(os.path.join(VF,"labels","video5_frame_labels.csv"))))
# label timeline: video_s of first in_fill frame per vial, and its wall-clock timestamp
lab={}
for r in frames:
    if r["in_fill"]=="1" and r["vial"] not in lab:
        lab[r["vial"]]=(float(r["video_s"]), r["timestamp"])
t0=datetime.fromisoformat(frames[0]["timestamp"])
print(f"detected fills: {len(fills)}   labelled vials: {len(lab)}")
print(f"\n{'det#':>4} {'det start (video_s)':>20} {'nearest label vial':>19} {'lbl start':>10} {'offset s':>9}")
offs=[]
for f in fills:
    ts=datetime.fromisoformat(f["start"]); vs=(ts-t0).total_seconds()
    best=min(lab.items(), key=lambda kv: abs(kv[1][0]-vs))
    off=vs-best[1][0]; offs.append(off)
    print(f"{f['fill']:>4} {vs:>20.1f} {best[0]:>19} {best[1][0]:>10.1f} {off:>9.1f}")
o=np.array(offs)
print(f"\noffset: median={np.median(o):+.1f}s  mean={o.mean():+.1f}s  sd={o.std():.1f}s  "
      f"range=[{o.min():+.1f},{o.max():+.1f}]")
