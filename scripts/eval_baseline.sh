#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

trap 'code=$?; echo; read -r -p "Exit code $code"; exit $code' EXIT

if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
  PY=".venv/Scripts/python.exe"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
elif command -v python >/devnull 2>&1; then
  PY="python"
else
  echo "Python not found.
Create a venv first:
  python -m venv .venv
  source .venv/bin/activate   # or .venv/Scripts/Activate.ps1 on Windows
  pip install -r requirements.txt"
  exit 1
fi

"$PY" -m src.evaluate --config configs/default.yaml --model baseline
