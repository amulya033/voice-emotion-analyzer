@echo off
echo =============================================
echo   Voice Emotion Analyzer - Setup
echo =============================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not on PATH.
    echo Please install Python from https://python.org and try again.
    pause
    exit /b 1
)

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing dependencies (this may take a few minutes)...
pip install torch transformers sounddevice numpy librosa --quiet

echo.
echo =============================================
echo   Setup complete! Run run.bat to start.
echo =============================================
pause
