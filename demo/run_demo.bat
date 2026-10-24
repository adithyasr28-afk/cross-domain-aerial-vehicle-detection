@echo off
setlocal
cd /d "%~dp0"
if "%DEMO_HOST%"=="" set "DEMO_HOST=127.0.0.1"
if "%DEMO_PORT%"=="" set "DEMO_PORT=7860"
py -3 app.py --server-name "%DEMO_HOST%" --server-port "%DEMO_PORT%"
if errorlevel 1 python app.py --server-name "%DEMO_HOST%" --server-port "%DEMO_PORT%"
endlocal
