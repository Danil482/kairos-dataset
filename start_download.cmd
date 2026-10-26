@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Python environment was not found in .venv.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -u scripts\start_download.py
pause
