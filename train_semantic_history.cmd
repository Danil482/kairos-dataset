@echo off
cd /d "%~dp0"
echo Preparing strictly earlier semantic author history...
.venv\Scripts\python.exe -u scripts\semantic_history.py
if errorlevel 1 goto failed
echo Training text-history models on fixed author folds...
.venv\Scripts\python.exe -u scripts\history_models.py --stage text
if errorlevel 1 goto failed
echo Adding the previously prepared image-content features...
.venv\Scripts\python.exe -u scripts\history_models.py --stage visual
if errorlevel 1 goto failed
.venv\Scripts\python.exe scripts\analyze_semantic_history.py
if errorlevel 1 goto failed
.venv\Scripts\python.exe scripts\plot_history_results.py
if errorlevel 1 goto failed
echo Complete. Results: results\semantic_history\comparison.json
pause
exit /b 0
:failed
echo Failed. Read the error above. Do not start duplicate runs.
pause
exit /b 1
