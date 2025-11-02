#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Keep window open on success or error 
trap 'code=$?; echo; read -r -p "Exit code $code"; exit $code' EXIT

# find/create venv
if [[ ! -x ".venv/bin/python" && ! -x ".venv/Scripts/python.exe" ]]; then
  echo "[setup] creating .venv ..."
  if command -v python3 >/dev/null 2>&1; then
    python3 -m venv .venv
  elif command -v py >/dev/null 2>&1; then
    py -3 -m venv .venv
  elif command -v python >/dev/null 2>&1; then
    python -m venv .venv
  else
    echo "[setup] Python not found. Install Python 3 first."
    exit 1
  fi
fi

# Pick venv python (Unix/Mac or Windows)
if [[ -x ".venv/bin/python" ]]; then
  VENV_PY=".venv/bin/python"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
  VENV_PY=".venv/Scripts/python.exe"
else
  echo "[setup] venv looks broken (no python found in .venv)."
  exit 1
fi

# install deps 
echo "[setup] upgrading pip/setuptools/wheel ..."
"$VENV_PY" -m pip install --upgrade pip setuptools wheel

if [[ -f "requirements.txt" ]]; then
  echo "[setup] installing requirements.txt ..."
  "$VENV_PY" -m pip install -r requirements.txt
else
  echo "[setup] requirements.txt not found"
  exit 1
fi

# Optional: download a spaCy model if spaCy is listed
if grep -qi '^spacy' requirements.txt; then
  "$VENV_PY" - <<'PY'
import sys, subprocess
try:
    import spacy
    spacy.load("en_core_web_sm")
    print("[setup] spaCy model already present.")
except Exception:
    print("[setup] downloading spaCy model (en_core_web_sm) ...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
PY
fi

echo "[setup] done."
case "$VENV_PY" in
  *.exe) echo "Activate: .\\.venv\\Scripts\\Activate.ps1 (PowerShell)  or  .\\.venv\\Scripts\\activate (cmd)";;
  *)     echo "Activate: source .venv/bin/activate";;
esac
