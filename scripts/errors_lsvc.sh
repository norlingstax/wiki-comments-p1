#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

trap 'code=$?; echo; read -r -p "Exit code $code"; exit $code' EXIT

PY=".venv/bin/python"
[[ -x "$PY" ]] || PY=".venv/Scripts/python.exe"
[[ -x "$PY" ]] || { echo "Missing virtualenv Python. Create .venv and install requirements."; read -r -p "Press Enter to close..."; exit 1; }

"$PY" -m src.interpret --config configs/default.yaml --model lsvc --threshold 0.0 --k 10
