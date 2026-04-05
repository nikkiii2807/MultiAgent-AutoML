$ErrorActionPreference = "Stop"

Write-Host "Starting Multi-Agent AutoML Stack in the Background..." -ForegroundColor Cyan

# Start Backend Job
Write-Host "Starting FastAPI Backend on port 8000..." -ForegroundColor Green
Start-Job -Name "AutoML_Backend" -ScriptBlock {
    cd "$using:PWD\backend"
    .\venv\Scripts\python.exe -m uvicorn main:app --port 8000
}

# Start Frontend Job
Write-Host "Starting Next.js Frontend on port 3000..." -ForegroundColor Green
Start-Job -Name "AutoML_Frontend" -ScriptBlock {
    cd "$using:PWD\frontend"
    npm run dev
}

Write-Host "Both servers started stealthily in the background." -ForegroundColor Yellow
Write-Host "Frontend is available at http://localhost:3000"
Write-Host "To stop them later, run: Get-Job | Remove-Job -Force" -ForegroundColor Gray
