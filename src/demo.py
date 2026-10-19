"""Show the Recipe 6 -> 7 carryover reading straight off the camera.

    python src/demo.py

Prints each detected fill across the changeover with the colour the measurement
produced, drawn as a true-colour block in the terminal.
"""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vialcolor import measure

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "data", "vial_fill", "images")
FILLS = os.path.join(ROOT, "out", "fills_video5.csv")

if not os.path.exists(FILLS):
    sys.exit("run `python src/segment_fills.py --video 5` first")

# Only emit colour escapes to a real terminal; piped or captured output gets hex.
TRUECOLOR = sys.stdout.isatty() and os.environ.get("TERM", "") not in ("", "dumb")

def block(r, g, b, w=10):
    if TRUECOLOR:
        return f"\033[48;2;{r};{g};{b}m{' ' * w}\033[0m"
    return f"#{r:02X}{g:02X}{b:02X}".ljust(w)

rows = list(csv.DictReader(open(FILLS)))
window = [r for r in rows if 4 <= int(r["fill"]) <= 10]

print("\n  Recipe 6 (orange) -> Recipe 7 (blue) changeover, measured from pixels only")
print("  " + "-" * 74)
print(f"  {'fill':>4}  {'swatch':<12} {'hue':>6} {'OD':>6} {'mL':>5}  {'RGB':>15}  note")
for r in window:
    rgb = (int(r["r"]), int(r["g"]), int(r["b"]))
    hue = float(r["hue_deg"])
    band = ("orange" if hue < 90 else
            "blue"   if 240 < hue < 300 else
            "green"  if 120 < hue < 200 else "teal")
    # Fill 7 is the first vial of Recipe 7; 5 and 6 are the Recipe 6 batch before it.
    note = band + ("   <- first vial of Recipe 7, should be BLUE" if r["fill"] == "7" else "")
    print(f"  {r['fill']:>4}  {block(*rgb)}  {hue:>6.1f} {float(r['od']):>6.3f} "
          f"{float(r['final_ml']):>5.0f}  {str(rgb):>15}  {note}")
print("  " + "-" * 74)
print("""
  Fills 5-6 are the Recipe 6 batch (orange). Fill 7 is the FIRST vial of Recipe 7,
  which doses 20 mL clear + 50 mL blue and uses no orange tank at all - yet it comes
  out green, because orange left in the shared mixing path is pushed into the vial
  ahead of the blue. Fills 8-9 run clean blue once the residue has flushed through.
  (Fill 4 is green legitimately: Recipe 5 doses all three tanks.)
""")

frame = window[3]["peak_frame"]
print(f"  Single-frame check on the green one ({frame}):")
m = measure(os.path.join(IMG, frame))
print(f"    RGB {m.rgb}   hue {m.hue_deg}   chroma {m.chroma}")
print(f"    absorbance R/G/B {m.absorbance}   level {m.level_ml} mL   confidence {m.confidence}")
print(f"\n  All 27 fills as one image:  open out/contact_sheet_video5.png\n")
