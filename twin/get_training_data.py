"""Render a labelled YOLO training set from the digital twin on spark7 and copy it to this PC.

    python get_training_data.py --n 3                  # quick check that it works (~1-2 min)
    python get_training_data.py --n 300                # the training set
    python get_training_data.py --n 300 --fast         # real-time renderer: much quicker, rougher light

The set lands in <repo>/data/twin_yolo/<date_time>/ (images/, labels/, meta.json).
"""
import argparse
import os
import time

from spark_common import HERE, REMOTE_DIR, connect, isaac_command, lines, start, sync_server_files

REPO = os.path.join(os.path.expanduser("~"), "MSOE", "Documents", "GitHub",
                    "msoe-ai-club-rockwell-automation-nvidia-hackath-michael-wood-s-team-1-2ccd0bfbe8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=int(time.time()) % 100)
    ap.add_argument("--fast", action="store_true", help="real-time renderer instead of path tracing")
    ap.add_argument("--subframes", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(REPO, "data", "twin_yolo"))
    a = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    remote_out = f"{REMOTE_DIR}/datasets/{stamp}"
    local_out = os.path.join(a.out, stamp)

    client = connect()
    sync_server_files(client)
    args = (f"--n {a.n} --seed {a.seed} --out {remote_out} --subframes {a.subframes}"
            + (" --renderer RaytracedLighting" if a.fast else ""))
    chan = start(client, isaac_command("synth_dataset.py", args))
    print(f"Rendering {a.n} labelled images on spark7 (first load takes a few minutes)...")
    log_path = os.path.join(HERE, "get_training_data.log")
    done, n = False, 0
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            for line in lines(chan):
                log.write(line + "\n")
                if line.startswith("SAVED"):
                    n += 1
                    print(f"  {n}/{a.n}  {os.path.basename(line)}")
                elif line.startswith(("FAILED", "Traceback")) or "Error:" in line:
                    print(" ", line)
                done = done or line.startswith("DONE")
    except KeyboardInterrupt:
        print("\nStopped; copying whatever finished.")
    finally:
        chan.close()

    sftp = client.open_sftp()
    copied = 0
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(local_out, sub), exist_ok=True)
        try:
            names = sftp.listdir(f"{remote_out}/{sub}")
        except FileNotFoundError:
            names = []
        for name in names:
            sftp.get(f"{remote_out}/{sub}/{name}", os.path.join(local_out, sub, name))
            copied += sub == "images"
    try:
        sftp.get(f"{remote_out}/meta.json", os.path.join(local_out, "meta.json"))
    except FileNotFoundError:
        pass
    sftp.close()
    client.close()

    print(f"\n{copied} labelled image(s) copied to {local_out}")
    if not done:
        print(f"The render did not finish cleanly; details in {log_path}")


if __name__ == "__main__":
    main()
