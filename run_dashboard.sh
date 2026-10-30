#!/usr/bin/env bash
# Recipe-Aware Color Monitor: install requirements if needed, then start the dashboard.
#   ./run_dashboard.sh              # pick the camera in the browser
#   ./run_dashboard.sh 0            # camera 0 (or a stream URL, video file or image folder)
#   ./run_dashboard.sh list         # snapshot every camera to out/camera_N.jpg
cd "$(dirname "$0")"
python3 -c "import cv2, numpy" 2>/dev/null || python3 -m pip install -r requirements.txt
if [ "$1" = "list" ]; then exec python3 src/dashboard.py --list-cameras; fi
if [ -n "$1" ]; then exec python3 src/dashboard.py --source "$1"; else exec python3 src/dashboard.py; fi
