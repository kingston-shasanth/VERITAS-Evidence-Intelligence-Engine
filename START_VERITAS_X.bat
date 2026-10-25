@echo off
setlocal enabledelayedexpansion

title VERITAS-X Launcher
color 0b

echo ==============================================================================
echo                      VERITAS-X CLAIM VERIFICATION PLATFORM                    
echo                 One-Click Windows Launcher & Runtime Manager                  
echo ==============================================================================
echo.

:: 1. Resolve project root directory reliably regardless of working directory
set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"
echo [*] Project Root: %ROOT_DIR%

:: 2. Detect Python Virtual Environment
set "PYTHON_EXE=%ROOT_DIR%.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo [!] Local virtual environment not found at .venv\Scripts\python.exe
    where python >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Python 3.10+ was not found in PATH or in .venv.
        echo         Please create a virtual environment: python -m venv .venv
        echo         and install dependencies: .venv\Scripts\pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
    set "PYTHON_EXE=python"
    echo [*] Falling back to system Python in PATH.
) else (
    echo [*] Detected Python Environment: %PYTHON_EXE%
)

:: 3. Check Node.js and NPM
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is not found in your system PATH.
    echo         Node.js v18+ is required to serve the React/Vite dashboard.
    echo         Please install Node.js from https://nodejs.org/
    echo.
    pause
    exit /b 1
)

where npm >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] npm is not found in your system PATH.
    echo.
    pause
    exit /b 1
)

:: 4. Verify Frontend node_modules
if not exist "%ROOT_DIR%ui\veritas-x-advanced-ui\node_modules\" (
    echo [*] Installing frontend dependencies in ui\veritas-x-advanced-ui...
    pushd "%ROOT_DIR%ui\veritas-x-advanced-ui"
    call npm install
    popd
)

:: 5. Port preflight check using helper script
echo [*] Performing network preflight check on ports 8011 (API) and 5173 (UI)...
for /f "tokens=1,2,3,4" %%A in ('"%PYTHON_EXE%" scripts\launcher_helper.py preflight-tokens') do (
    set "API_OCCUPIED=%%A"
    set "API_HEALTHY=%%B"
    set "FE_OCCUPIED=%%C"
    set "FE_READY=%%D"
)


:: 6. Launch Backend Server if not already running healthy
if "%API_HEALTHY%"=="True" (
    echo [+] Backend API is already running and healthy on http://127.0.0.1:8011
) else if "%API_OCCUPIED%"=="True" (
    echo [WARNING] Port 8011 is occupied by an unrecognized process.
    echo           Please close the process or run STOP_VERITAS_X.bat first.
    echo.
    pause
    exit /b 1
) else (
    echo [*] Starting VERITAS-X Python API Server on http://127.0.0.1:8011...
    start "VERITAS-X Backend API Server [Port 8011]" cmd /k "cd /d "%ROOT_DIR%" && "%PYTHON_EXE%" -m api.server --port 8011 --background-warmup"
)

:: 7. Launch Frontend Vite Server if not already running
if "%FE_READY%"=="True" (
    echo [+] Frontend dashboard is already active on http://127.0.0.1:5173
) else if "%FE_OCCUPIED%"=="True" (
    echo [WARNING] Port 5173 is occupied by another process.
    echo.
    pause
    exit /b 1
) else (
    echo [*] Starting React/Vite Frontend on http://127.0.0.1:5173...
    start "VERITAS-X React Frontend [Port 5173]" cmd /k "cd /d "%ROOT_DIR%ui\veritas-x-advanced-ui" && npm run dev"
)

:: 8. Wait for Readiness
echo.
"%PYTHON_EXE%" scripts\launcher_helper.py wait-ready
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] One or more services failed to initialize within the timeout.
    echo         Please inspect the Backend and Frontend console windows for error traces.
    echo.
    pause
    exit /b 1
)

:: 9. Open Web Browser
echo.
echo ==============================================================================
echo   SUCCESS: VERITAS-X Evidence Verification Platform is ONLINE!
echo   ----------------------------------------------------------------------------
echo   Dashboard UI : http://127.0.0.1:5173/
echo   Backend API  : http://127.0.0.1:8011/api/health
echo ==============================================================================
echo.
echo [*] Launching application in your default web browser...
start http://127.0.0.1:5173

echo.
echo [*] Press any key or run STOP_VERITAS_X.bat when you wish to close this window.
echo     (To stop servers safely, run STOP_VERITAS_X.bat or press Ctrl+C in their terminals.)
echo.
pause
