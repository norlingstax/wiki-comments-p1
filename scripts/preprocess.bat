@echo off
setlocal
pushd "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
  echo Could not find .venv\Scripts\python.exe in "%CD%".
  echo Create/activate your venv first:  python -m venv .venv
  echo Then install deps:               pip install -r requirements.txt
  pause
  exit /b 1
)

echo Running preprocessing...
".venv\Scripts\python.exe" -m src.preprocess --config configs\default.yaml

set RC=%ERRORLEVEL%
if not "%RC%"=="0" echo Preprocessing failed with exit code %RC%.
pause
exit /b %RC%