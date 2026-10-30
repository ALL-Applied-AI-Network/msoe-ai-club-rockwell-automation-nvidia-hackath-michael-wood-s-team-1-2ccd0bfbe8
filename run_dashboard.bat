@echo off
title Recipe-Aware Color Monitor
cd /d "%~dp0"
python -c "import cv2, numpy" 2>nul || (
  echo Installing requirements...
  python -m pip install -r requirements.txt
)
echo.
echo Camera: a number (0, 1, ...), an rtsp:// or http:// stream, a video file, or a folder of images.
echo Leave blank to pick it in the browser. Type "list" to snapshot every camera first.
set "SRC="
set /p "SRC=Source: "
if /i "%SRC%"=="list" (
  python src\dashboard.py --list-cameras
  set "SRC="
  set /p "SRC=Source: "
)
if "%SRC%"=="" (
  python src\dashboard.py
) else (
  python src\dashboard.py --source "%SRC%"
)
pause
