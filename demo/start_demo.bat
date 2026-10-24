@echo off
setlocal
cd /d "%~dp0"
set GRADIO_ANALYTICS_ENABLED=False
set YOLO_OFFLINE=True
set YOLO_AUTOINSTALL=False
set HF_HUB_OFFLINE=1
set HF_HUB_DISABLE_TELEMETRY=1
set PYTHONIOENCODING=utf-8
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" launch_check.py %*
) else (
  py -3 --version >nul 2>&1
  if not errorlevel 1 (
    py -3 launch_check.py %*
  ) else (
    python --version >nul 2>&1
    if errorlevel 1 (
      echo ERROR: Python is missing or is a Microsoft Store alias. Install Python and follow README.md.
      pause
      exit /b 1
    )
    python launch_check.py %*
  )
)
if errorlevel 1 (
  echo ERROR: Demo could not start. See the message above and README.md. No packages are downloaded by this launcher.
  pause
  exit /b 1
)
