@echo off
rem AnySport hub – Windows launcher
cd /d "%~dp0hub"
if not exist .venv (
  py -3 -m venv .venv || python -m venv .venv
  .venv\Scripts\pip install -r requirements.txt
)
.venv\Scripts\python -m anysport
pause
