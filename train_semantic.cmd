@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -u scripts\semantic_embeddings.py --download-only
if errorlevel 1 goto end
".venv\Scripts\python.exe" -u scripts\semantic_embeddings.py --batch-size 32
if errorlevel 1 goto end
".venv\Scripts\python.exe" -u scripts\semantic_models.py
:end
pause
