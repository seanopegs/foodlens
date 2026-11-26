@echo off
setlocal
title FoodLens
cd /d "%~dp0"

rem Find Python: the "py" launcher first, then "python".
set "PY="
py -c "import sys" >nul 2>&1 && set "PY=py"
if not defined PY (
    python -c "import sys" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo Python was not found.
    echo Install Python 3.10 or newer from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" during setup, then run this file again.
    echo.
    pause
    exit /b 1
)

rem Install the packages the first time only.
%PY% -c "import streamlit, torch, torchvision, PIL" >nul 2>&1
if errorlevel 1 (
    echo Installing the packages FoodLens needs. This only happens once and can take a few minutes...
    echo.
    %PY% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Installing the packages failed. See the messages above.
        pause
        exit /b 1
    )
)

if not exist "checkpoints\best.pt" (
    echo The model file checkpoints\best.pt is missing.
    echo Download or pull the full repository and run this file again.
    echo.
    pause
    exit /b 1
)

echo Starting FoodLens. Your browser opens at http://localhost:8501
echo To stop the app, close this window or press Ctrl+C.
echo.
%PY% -m streamlit run app.py
if errorlevel 1 pause
endlocal
