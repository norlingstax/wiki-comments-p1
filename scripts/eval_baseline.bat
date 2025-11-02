@echo off
setlocal
pushd "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv\Scripts\python.exe
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -m src.evaluate --config configs/default.yaml --model baseline

set RC=%ERRORLEVEL%
if not "%RC%"=="0" echo Evaluation failed with exit code %RC%.
pause
exit /b %RC%
