#!/usr/bin/env bash
# Recipe-Aware Color Monitor: set up on first run (bash setup.sh), then start the dashboard.
#   ./run_dashboard.sh              # pick the camera in the browser
#   ./run_dashboard.sh 0            # camera 0 (or a stream URL, video file or image folder)
#   ./run_dashboard.sh list         # snapshot every camera to out/camera_N.jpg
cd "$(dirname "$0")"
if ! .venv/bin/python -c "import cv2, numpy" 2>/dev/null; then
    bash setup.sh || exit 1
fi
PY=.venv/bin/python
if [ "$1" = "list" ]; then exec "$PY" src/dashboard.py --list-cameras; fi
if [ -n "$1" ]; then exec "$PY" src/dashboard.py --source "$1"; else exec "$PY" src/dashboard.py; fi
