@echo off
set ROOT=%~dp0
if not exist "%ROOT%.venv\Scripts\python.exe" (
    python -m venv "%ROOT%.venv"
)
"%ROOT%.venv\Scripts\python.exe" -m pip install --upgrade pip
"%ROOT%.venv\Scripts\python.exe" -m pip install -r "%ROOT%backend\requirements-advanced.txt"
pause
