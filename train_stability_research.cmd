@echo off
cd /d "%~dp0"
echo Training seed stability, joint content and decision-rule experiments...
.venv\Scripts\python.exe -u scripts\stability_models.py
if errorlevel 1 goto failed
.venv\Scripts\python.exe scripts\report_stability.py
if errorlevel 1 goto failed
echo Complete. Results: results\stability_research\comparison.json
pause
exit /b 0
:failed
echo Failed. Read the error above. Do not start duplicate runs.
pause
exit /b 1
