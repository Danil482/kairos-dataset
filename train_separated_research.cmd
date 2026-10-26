@echo off
cd /d "%~dp0"
echo Training disjoint stopping/decision and E5 capacity experiments...
.venv\Scripts\python.exe -u scripts\separated_models.py
if errorlevel 1 goto failed
.venv\Scripts\python.exe scripts\report_separated.py
if errorlevel 1 goto failed
echo Complete. Results: results\separated_research\comparison.json
pause
exit /b 0
:failed
echo Failed. Read the error above. Do not start duplicate runs.
pause
exit /b 1
