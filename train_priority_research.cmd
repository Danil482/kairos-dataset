@echo off
cd /d "%~dp0"
echo Preparing controlled v2 experiments...
.venv\Scripts\python.exe -u scripts\priority_features.py
if errorlevel 1 goto failed
.venv\Scripts\python.exe -u scripts\priority_models.py
if errorlevel 1 goto failed
.venv\Scripts\python.exe scripts\analyze_priority_research.py
if errorlevel 1 goto failed
.venv\Scripts\python.exe scripts\check_priority_artifacts.py
if errorlevel 1 goto failed
.venv\Scripts\python.exe scripts\plot_priority_results.py
if errorlevel 1 goto failed
.venv\Scripts\python.exe scripts\write_priority_report.py
if errorlevel 1 goto failed
echo Complete. Results: results\priority_research\comparison.json
pause
exit /b 0
:failed
echo Failed. Read the error above. Do not run duplicate training processes.
pause
exit /b 1
