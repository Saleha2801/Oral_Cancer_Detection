Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "   Oral Cancer AI: Multimodal Detection System" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan

Set-Location $PSScriptRoot

Write-Host "[1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ..." -ForegroundColor Green
Start-Process -FilePath "$PSScriptRoot\.venv\Scripts\python.exe" -ArgumentList "-m uvicorn backend.main:app --host 127.0.0.1 --port 8000" -WindowStyle Minimized

Start-Sleep -Seconds 2

Write-Host "[2/2] Opening Web Application in Browser..." -ForegroundColor Green
Start-Process "http://localhost:5173"

Write-Host "Starting Vite Frontend..." -ForegroundColor Yellow
npm run dev
