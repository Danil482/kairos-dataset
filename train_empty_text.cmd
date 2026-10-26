@echo off
cd /d "%~dp0"
echo Auditing posts without original text...
.venv\Scripts\python.exe scripts\empty_text_audit.py
if errorlevel 1 goto failed
.venv\Scripts\python.exe scripts\ocr_empty_posts.py --download-only
if errorlevel 1 goto failed
echo Training diagnostic variants...
.venv\Scripts\python.exe -u scripts\empty_text_models.py --stage base
if errorlevel 1 goto failed
echo Reading text from local content images...
.venv\Scripts\python.exe -u scripts\ocr_empty_posts.py
if errorlevel 1 goto failed
.venv\Scripts\python.exe -u scripts\empty_text_models.py --stage ocr
if errorlevel 1 goto failed
.venv\Scripts\python.exe scripts\plot_empty_text_results.py
if errorlevel 1 goto failed
echo Complete. Results: results\empty_text\comparison.json
pause
exit /b 0
:failed
echo Failed. Read the error above. Do not start duplicate runs.
pause
exit /b 1
