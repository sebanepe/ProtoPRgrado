# Run full test suite, produce reports (junit xml + coverage).
# Activate venv if present, then run pytest with common reporting flags.

Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -ErrorAction SilentlyContinue
$venv = Join-Path -Path (Get-Location) -ChildPath ".\.venv\Scripts\Activate.ps1"
if (Test-Path $venv) {
    Write-Host "Activating virtualenv..."
    & $venv
} else {
    Write-Host "Virtualenv activation script not found. Ensure you activated .venv manually if needed."
}

$reports = Join-Path -Path (Get-Location) -ChildPath "reports"
if (-not (Test-Path $reports)) { New-Item -ItemType Directory -Path $reports | Out-Null }

Write-Host "Running pytest (junit xml + coverage)..."
python -m pytest backend/app/tests --junitxml=$reports\junit.xml --cov=backend --cov-report=term-missing -q

Write-Host "Attempting HTML report (requires pytest-html). If plugin missing, this will be skipped." 
python -m pytest backend/app/tests --html=$reports\report.html -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "pytest-html not available or HTML generation failed; skipping HTML report."
}

Write-Host "Reports available in: $reports"

# Run frontend unit tests if Node and npm are available
$frontendPkg = Join-Path -Path (Get-Location) -ChildPath "frontend\package.json"
if (Test-Path $frontendPkg) {
    if (Get-Command npm -ErrorAction SilentlyContinue) {
        Write-Host "Installing frontend dev dependencies..."
        Push-Location frontend
        npm ci --silent
        if ($LASTEXITCODE -ne 0) {
            Write-Host "npm ci failed, attempting npm install..."
            npm install --silent
        }
        Write-Host "Running frontend unit tests (vitest)..."
        npm run test --silent
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Frontend tests failed or vitest not configured"
        }
        Pop-Location
    } else {
        Write-Host "npm not found - skipping frontend tests"
    }
}
