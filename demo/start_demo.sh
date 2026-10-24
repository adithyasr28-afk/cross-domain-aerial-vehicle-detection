#!/usr/bin/env bash
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export GRADIO_ANALYTICS_ENABLED=False YOLO_OFFLINE=True YOLO_AUTOINSTALL=False
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONIOENCODING=utf-8
if [ -x .venv/bin/python ]; then
  PYTHON=.venv/bin/python
elif command -v python3 >/dev/null 2>&1 && python3 --version >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1 && python --version >/dev/null 2>&1; then
  PYTHON=python
else
  echo 'ERROR: Python is missing. Install Python and follow README.md. No internet is used by this launcher.' >&2
  exit 1
fi
exec "$PYTHON" launch_check.py "$@"
