@echo off
echo ========================================
echo Starting Connect 4 AI Setup...
echo ========================================

:: Check if the virtual environment folder exists
if not exist "venv" (
    echo Creating an isolated Python environment...
    python -m venv venv
)

:: Activate the environment
call venv\Scripts\activate.bat

:: Install the required libraries quietly
echo Downloading game libraries (this takes a minute the first time)...
python -m pip install --upgrade pip >nul 2>&1
pip install torch numpy pygame >nul 2>&1

:: Run the engine
echo Launching the arena...
python connect4_engine.py

:: Keep the window open if the game crashes so they can read the error
pause