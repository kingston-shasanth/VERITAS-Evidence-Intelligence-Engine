@echo off
setlocal enabledelayedexpansion

title VERITAS-X Shutdown Utility
color 0c

echo ==============================================================================
echo                      VERITAS-X CLAIM VERIFICATION PLATFORM                    
echo                         Safe Application Shutdown                             
echo ==============================================================================
echo.

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

set "PYTHON_EXE=%ROOT_DIR%.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" scripts\launcher_helper.py stop

echo.
echo ==============================================================================
echo   All VERITAS-X server processes have been safely stopped.
echo ==============================================================================
echo.
pause
