"""Render a labelled YOLO training set from the fill-station twin (domain-randomised).

    synth_dataset.py --n 300 --out ~/digital_twin/datasets/run1 [--renderer RaytracedLighting]

Each sample picks a random moment of the fill animation, a random liquid colour and
opacity (orange/blue/clear blends like the real recipes), and random light levels, then
writes images/<i>.png and labels/<i>.txt in YOLO format. Boxes come from Replicator's
tight 2-D bounding boxes, so labels are exact and free. Classes: 0 vial, 1 liquid,
2 nozzle. Prints "SAVED <path>" per image and "DONE" at the end.
"""
import argparse, json, os, random, traceback
from pathlib import Path

TWIN = os.path.expanduser("~/digital_twin/csi_digital_twin_v1")
STAGE = os.path.join(TWIN, "stations/fill_station/fill_station.usda")
ROOT = "/FillStation/FillStation_LocalRoot"
LIQUID_SHADER = "/FillStation/_materials/USD_blue_liquid_approximation/Principled_BSDF"
CLASSES = ["vial", "liquid", "nozzle"]
LABELLED = {f"{ROOT}/Fill_vial_active": "vial", f"{ROOT}/Fill_liquid": "liquid",
            f"{ROOT}/Fill_nozzle_stainless": "nozzle"}
# Tank dye colours as linear RGB transmittance; T-100 is clear. Blends cover the recipes.
TANKS = {"T-100": (0.95, 0.95, 0.95), "T-200": (0.85, 0.25, 0.03), "T-400": (0.006, 0.2, 0.3)}
LAST_FRAME = 686

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--renderer", default="PathTracing", choices=["PathTracing", "RaytracedLighting"])
ap.add_argument("--subframes", type=int, default=16)
a = ap.parse_args()
random.seed(a.seed)
out = Path(a.out).expanduser().resolve()
(out / "images").mkdir(parents=True, exist_ok=True)
(out / "labels").mkdir(parents=True, exist_ok=True)
cam = json.load(open(os.path.join(TWIN, "cameras", "FillCam.json")))
W, H = cam["render_px"]

from isaacsim import SimulationApp
kit = SimulationApp({"headless": True, "renderer": a.renderer, "width": W, "height": H})
try:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("omni.replicator.core")
    import numpy as np, cv2
    import omni.usd, omni.timeline
    import omni.replicator.core as rep
    from pxr import Gf, Usd, UsdGeom

    ctx = omni.usd.get_context()
    if not ctx.open_stage(STAGE):
        raise RuntimeError(f"could not open stage: {STAGE}")
    while ctx.get_stage_loading_status()[2] > 0:
        kit.update()
    for _ in range(20):
        kit.update()
    stage = ctx.get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())  # never write into the twin files
    tl = omni.timeline.get_timeline_interface()
    tps = stage.GetTimeCodesPerSecond()
    rep.orchestrator.set_capture_on_play(False)

    try:
        from isaacsim.core.utils.semantics import add_labels
        label = lambda prim, c: add_labels(prim, labels=[c], instance_name="class")
    except ImportError:
        from isaacsim.core.utils.semantics import add_update_semantics
        label = lambda prim, c: add_update_semantics(prim, c)
    for path, c in LABELLED.items():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"missing prim {path}")
        label(prim, c)

    shader = stage.GetPrimAtPath(LIQUID_SHADER)
    lights = [(p, p.GetAttribute("inputs:intensity"), p.GetAttribute("inputs:intensity").Get())
              for p in stage.Traverse() if "Light" in p.GetTypeName()]

    cam_prim = cam["prim_in_station_file"]
    rp = rep.create.render_product(cam_prim, (W, H))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    bbox = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
    rgb.attach([rp]); bbox.attach([rp])
    kit.update()
    # Same fix as batch_render.py: force RTX to re-sync the camera clipping range.
    clip = UsdGeom.Camera(stage.GetPrimAtPath(cam_prim)).GetClippingRangeAttr()
    near_far = clip.Get()
    clip.Set(Gf.Vec2f(near_far[0] * 0.5, near_far[1])); kit.update()
    clip.Set(near_far); kit.update()

    def capture():
        data = None
        for i in range(22):
            rep.orchestrator.step(rt_subframes=a.subframes, delta_time=0.0)
            data = np.asarray(rgb.get_data())
            if i >= 1 and data.ndim == 3 and data.size and data[..., :3].mean() > 1.0:
                return data[..., :3].copy(), bbox.get_data()
        raise RuntimeError("no image from the renderer")

    W2C = np.linalg.inv(np.array(cam["camera_to_world_in_station_file"]))
    K = np.array(cam["intrinsics_K_px"])
    liquid_prim = stage.GetPrimAtPath(f"{ROOT}/Fill_liquid/Fill_liquid")

    def liquid_box(frame):
        """Pixel box of the liquid at `frame`, or None while it is hidden."""
        t = Usd.TimeCode(frame)
        if UsdGeom.Imageable(liquid_prim).ComputeVisibility(t) == UsdGeom.Tokens.invisible:
            return None
        rng = UsdGeom.BBoxCache(t, ["default", "render"], useExtentsHint=False) \
            .ComputeWorldBound(liquid_prim).ComputeAlignedRange()
        if rng.IsEmpty():
            return None
        lo, hi = rng.GetMin(), rng.GetMax()
        us, vs = [], []
        for x in (lo[0], hi[0]):
            for y in (lo[1], hi[1]):
                for z in (lo[2], hi[2]):
                    X, Y, Z = (W2C @ np.array([x, y, z, 1.0]))[:3]    # USD camera looks down -Z
                    us.append(K[0, 0] * X / -Z + K[0, 2])
                    vs.append(K[1, 2] - K[1, 1] * Y / -Z)
        x0, y0 = max(0.0, min(us)), max(0.0, min(vs))
        x1, y1 = min(W - 1.0, max(us)), min(H - 1.0, max(vs))
        return (x0, y0, x1, y1) if x1 - x0 > 2 and y1 - y0 > 2 else None

    meta = []
    for i in range(a.n):
        # 10% before the liquid appears (frame < 21): negatives for the liquid class.
        frame = random.randint(1, 20) if random.random() < 0.1 else random.randint(21, LAST_FRAME)
        w = [random.random() ** 2 for _ in TANKS]
        w = [x / sum(w) for x in w]
        col = [sum(wi * TANKS[t][k] for wi, t in zip(w, TANKS)) for k in range(3)]
        col = [min(1.0, max(0.0, c * random.uniform(0.8, 1.2))) for c in col]
        opacity = random.uniform(0.35, 0.85)
        shader.GetAttribute("inputs:diffuseColor").Set(Gf.Vec3f(*col))
        shader.GetAttribute("inputs:opacity").Set(opacity)
        gain = random.uniform(0.6, 1.4)
        for _, attr, base in lights:
            attr.Set(base * gain * random.uniform(0.85, 1.15))
        tl.set_current_time(frame / tps)
        kit.update()

        img, bb = capture()
        id2lab = {int(k): v.get("class", "") for k, v in bb["info"]["idToLabels"].items()}
        rows = []
        for r in bb["data"]:
            c = id2lab.get(int(r["semanticId"]), "")
            if c not in CLASSES or r["x_max"] <= r["x_min"] or r["y_max"] <= r["y_min"]:
                continue
            x0, y0 = max(0, r["x_min"]), max(0, r["y_min"])
            x1, y1 = min(W - 1, r["x_max"]), min(H - 1, r["y_max"])
            rows.append(f"{CLASSES.index(c)} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} "
                        f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
        # The tight-box annotator drops see-through materials, so the liquid gets no box from
        # it. Project the liquid mesh's world bounds at this frame through FillCam instead.
        if not any(r.startswith("1 ") for r in rows):
            b = liquid_box(frame)
            if b:
                x0, y0, x1, y1 = b
                rows.append(f"1 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} "
                            f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
        name = f"twin_{a.seed:02d}_{i:05d}"
        cv2.imwrite(str(out / "images" / f"{name}.png"), img[..., ::-1])
        (out / "labels" / f"{name}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        meta.append(dict(name=name, frame=frame, color=col, opacity=opacity, light_gain=gain,
                         tank_weights=dict(zip(TANKS, w)), boxes=len(rows)))
        print(f"SAVED {out / 'images' / name}.png  boxes={len(rows)}", flush=True)

    (out / "meta.json").write_text(json.dumps(dict(classes=CLASSES, samples=meta), indent=1))
    print("DONE", flush=True)
except Exception:
    traceback.print_exc()
    print("FAILED", flush=True)
finally:
    kit.close()
