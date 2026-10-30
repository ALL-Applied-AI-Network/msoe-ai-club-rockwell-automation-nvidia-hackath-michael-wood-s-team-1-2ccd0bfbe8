#!/usr/bin/env bash
# One-command install + self-test for Ubuntu / macOS. No sudo needed.
#
#   bash setup.sh          # creates .venv, installs requirements, runs src/selftest.py
#
# A fresh Ubuntu often lacks the python3-venv / python3-pip packages, and installing them
# needs sudo. If the normal venv cannot get pip, this makes the venv without pip and installs
# pip into it with PyPA's official bootstrap (https://bootstrap.pypa.io/get-pip.py), all inside
# this folder. Nothing outside the repository is changed.
set -u
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || { echo "ERROR: $PY not found. Install Python 3.10+."; exit 1; }
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || {
    echo "ERROR: $("$PY" --version 2>&1) is too old; need Python 3.10 or newer."; exit 1; }
echo "using $("$PY" --version 2>&1) at $(command -v "$PY")"

if [ ! -x .venv/bin/python ] || ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    rm -rf .venv
    if "$PY" -m venv .venv >/dev/null 2>&1 && .venv/bin/python -m pip --version >/dev/null 2>&1; then
        echo "created .venv"
    else
        echo "python3-venv / ensurepip missing: creating .venv without pip and bootstrapping pip"
        rm -rf .venv
        "$PY" -m venv --without-pip .venv || { echo "ERROR: cannot create a virtual environment"; exit 1; }
        GETPIP=.venv/get-pip.py
        if command -v curl >/dev/null 2>&1; then
            curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$GETPIP"
        elif command -v wget >/dev/null 2>&1; then
            wget -q https://bootstrap.pypa.io/get-pip.py -O "$GETPIP"
        else
            .venv/bin/python -c "import urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', '$GETPIP')"
        fi
        [ -s "$GETPIP" ] || { echo "ERROR: could not download get-pip.py (no internet?)"; exit 1; }
        .venv/bin/python "$GETPIP" --quiet || { echo "ERROR: pip bootstrap failed"; exit 1; }
        rm -f "$GETPIP"
    fi
fi

.venv/bin/python -m pip install --quiet --disable-pip-version-check -r requirements.txt || {
    echo "ERROR: package install failed (see above). Check internet / proxy."; exit 1; }
echo "installed: $(.venv/bin/python -c 'import cv2, numpy; print("OpenCV", cv2.__version__, "NumPy", numpy.__version__)')"

.venv/bin/python src/selftest.py "$@"
status=$?
if [ $status -eq 0 ]; then
    echo
    echo "Ready. Start the dashboard with:"
    echo "  .venv/bin/python src/dashboard.py --list-cameras"
    echo "  .venv/bin/python src/dashboard.py --source <camera number>"
fi
exit $status
