@echo off
chcp 65001 > nul
echo ===================================================
echo       NKUST AoV Tool - Startup Script
echo ===================================================
echo.

:: 1. Check for Virtual Environment
if exist .venv (
    echo [Info] Activating virtual environment (.venv)...
    call .venv\Scripts\activate
) else (
    echo [Info] No .venv found. Using system Python.
)

:: 2. Check for Dependencies (Optional, keeps it fast)
:: python -c "import streamlit" 2>nul
:: if %errorlevel% neq 0 (
::     echo [Error] Streamlit not found. Installing requirements...
::     pip install -r requirements.txt
:: )

:: 3. Run Application
echo.
echo [Info] Launching Application...
echo.
streamlit run aov_app.py

if %errorlevel% neq 0 (
    echo.
    echo [Error] Application failed to start.
    echo Common fixes:
    echo  1. pip install -r requirements.txt
    echo  2. Ensure Graphviz is installed (choco install graphviz)
    echo.
    pause
)
