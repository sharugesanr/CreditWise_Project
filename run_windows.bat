@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10-3.12 and enable Add Python to PATH.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" python -m venv .venv
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed. Check your internet connection and Python version.
  pause
  exit /b 1
)
python -m streamlit run app.py
pause
