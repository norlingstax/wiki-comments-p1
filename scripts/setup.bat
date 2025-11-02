@echo off
setlocal
REM Go to repo root (this file sits in scripts\)
pushd "%~dp0.."

REM Create venv if missing (use Windows 'py' launcher if available)
if not exist ".venv\Scripts\python.exe" (
  echo [setup] creating .venv ...
  py -m venv .venv || python -m venv .venv
)

REM Use the venv's python for everything
set VENV_PY=.venv\Scripts\python.exe

REM Upgrade pip + install requirements
"%VENV_PY%" -m pip install --upgrade pip wheel setuptools
if exist requirements.txt (
  echo [setup] installing from requirements.txt ...
  "%VENV_PY%" -m pip install -r requirements.txt
) else (
  echo requirements.txt not found
  exit /b 1
)

REM If spaCy is listed, download the English model once
REM    findstr sets ERRORLEVEL=0 when it finds a match
findstr /I /R "^spacy" requirements.txt >nul
if "%ERRORLEVEL%"=="0" (
  echo [setup] downloading spaCy model (en_core_web_sm) ...
  "%VENV_PY%" -m spacy download en_core_web_sm
)

echo [setup] done. To activate: .venv\Scripts\activate
popd
exit /b 0
