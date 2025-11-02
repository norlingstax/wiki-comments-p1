@echo off
setlocal
pushd "%~dp0\.."

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo Missing virtualenv Python. Create .venv and install requirements.
  pause
  exit /b 1
)

"%PY%" -m src.interpret --config configs\default.yaml --model pac --threshold 0.0

set RC=%ERRORLEVEL%
if not "%RC%"=="0" echo Script failed with exit code %RC%.
pause
exit /b %RC%
