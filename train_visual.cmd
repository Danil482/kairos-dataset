@echo off
cd /d "%~dp0"
echo Preparing local visual encoder...
.venv\Scripts\python.exe scripts\visual_embeddings.py --download-only
if errorlevel 1 goto failed
echo Extracting image content vectors. This stage can resume after interruption.
.venv\Scripts\python.exe scripts\visual_embeddings.py
if errorlevel 1 goto failed
echo Training visual models on the fixed author folds...
.venv\Scripts\python.exe scripts\visual_models.py
if errorlevel 1 goto failed
.venv\Scripts\python.exe scripts\plot_visual_results.py
if errorlevel 1 goto failed
echo Complete. Results: results\visual\comparison.json
pause
exit /b 0
:failed
echo Failed. Read the error above before restarting. Do not start duplicate runs.
pause
exit /b 1
