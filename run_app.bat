@echo off
echo =========================================================
echo    Oral Cancer AI: Multimodal Detection System
echo =========================================================
cd /d "%~dp0"

echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "Oral Cancer Backend" "%~dp0.venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

timeout /t 2 /nobreak >nul

echo [2/2] Opening Web App at http://localhost:5173 ...
start http://localhost:5173

echo Starting Vite Frontend Server...
npm run dev
