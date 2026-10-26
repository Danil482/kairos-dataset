@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe -u scripts\evaluate_frozen_test.py --predict-only
if errorlevel 1 goto failed
.venv\Scripts\python.exe -u scripts\audit_test_inference.py
if errorlevel 1 goto failed
.venv\Scripts\python.exe -u scripts\evaluate_frozen_test.py
if errorlevel 1 goto failed
echo Complete. Results: results\official_test_v3\test_metrics.json
pause
exit /b 0
:failed
echo Evaluation failed. Read the error above. Saved predictions are preserved.
pause
exit /b 1
