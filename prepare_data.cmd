@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -u scripts\prepare_data.py
pause
