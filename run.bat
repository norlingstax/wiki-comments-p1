@echo off
setlocal
REM cd to repo root
pushd "%~dp0\.."

REM Which models to run:
REM   scripts\run.bat           -> baseline + logreg + pac + lsvc (default)
REM   scripts\run.bat baseline  -> baseline only (evaluate)
REM   scripts\run.bat logreg    -> logreg only
REM   scripts\run.bat pac       -> pac only
REM   scripts\run.bat lsvc      -> lsvc only
REM   scripts\run.bat all       -> baseline + logreg + pac + lsvc
set "MODELS=%~1"
if "%MODELS%"=="" set "MODELS=all"

REM Ensure venv
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo Creating .venv ...
  py -m venv .venv
)
if not exist "%PY%" (
  echo Could not find virtualenv Python
  pause
  exit /b 1
)

REM Install deps
echo Installing requirements ...
"%PY%" -m pip install -U pip >nul
"%PY%" -m pip install -r requirements.txt

REM Uncomment this for spaCy resources
REM "%PY%" -m spacy download en_core_web_sm

echo.
echo ===== PREPROCESS: build/cache features =====
"%PY%" -m src.preprocess --config configs\default.yaml
if errorlevel 1 goto :fail

if /I "%MODELS%"=="baseline" (
  call :run_baseline
  goto :end
) else if /I "%MODELS%"=="logreg" (
  call :run_classifier logreg
  goto :end
) else if /I "%MODELS%"=="pac" (
  call :run_classifier pac
  goto :end
) else if /I "%MODELS%"=="lsvc" (
  call :run_classifier lsvc
  goto :end
) else if /I "%MODELS%"=="all" (
  call :run_baseline
  call :run_classifier logreg
  call :run_classifier pac
  call :run_classifier lsvc
  goto :end
) else (
  echo Unknown arg "%MODELS%" (use: baseline ^| logreg ^| pac ^| lsvc ^| all)
  pause
  exit /b 2
)

goto :end

:run_baseline
echo.
echo ===== BASELINE: evaluate only =====
"%PY%" -m src.evaluate --config configs\default.yaml --model baseline
if errorlevel 1 goto :fail
exit /b 0

:run_classifier
set "M=%~1"
if /I "%M%"=="baseline" (
  echo Refusing to train/interpret baseline (no classifier).
  exit /b 0
)
echo.
echo ===== MODEL: %M% =====
echo [1/3] Training...
"%PY%" -m src.train --config configs\default.yaml --model %M%
if errorlevel 1 goto :fail
echo [2/3] Evaluating on TEST...
"%PY%" -m src.evaluate --config configs\default.yaml --model %M%
if errorlevel 1 goto :fail
echo [3/3] Interpreting (FP/FN from VAL)...
"%PY%" -m src.interpret --config configs\default.yaml --model %M% --k 3
if errorlevel 1 goto :fail
exit /b 0

:fail
echo Script failed.
pause
exit /b 1

:end
pause
