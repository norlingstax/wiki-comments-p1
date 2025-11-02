#!/usr/bin/env bash
set -euo pipefail

# cd to repo root
cd "$(dirname "$0")/.."

# Which models to run:
#   ./scripts/run.sh            -> baseline + logreg + pac (default)
#   ./scripts/run.sh baseline   -> baseline only (evaluate)
#   ./scripts/run.sh logreg     -> logreg only
#   ./scripts/run.sh pac        -> pac only
#   ./scripts/run.sh lsvc       -> lsvc only
#   ./scripts/run.sh all        -> baseline + logreg + pac + lsvc
MODELS="${1:-all}"

# Ensure virtualenv
if [[ ! -x ".venv/bin/python" && ! -x ".venv/Scripts/python.exe" ]]; then
  echo "Creating .venv ..."
  python -m venv .venv
fi
PY=".venv/bin/python"
[[ -x "$PY" ]] || PY=".venv/Scripts/python.exe"
[[ -x "$PY" ]] || { echo "Could not find virtualenv Python!"; exit 1; }

# Install deps
echo "Installing requirements ..."
"$PY" -m pip install -U pip >/dev/null
"$PY" -m pip install -r requirements.txt

# Uncomment this for spaCy resources
# "$PY" -m spacy download en_core_web_sm

echo
echo "===== PREPROCESS: build/cache features ====="
"$PY" -m src.preprocess --config configs/default.yaml

run_baseline () {
  echo ""
  echo "===== BASELINE: evaluate only ====="
  "$PY" -m src.evaluate --config configs/default.yaml --model baseline
}

run_classifier () {
  local M="$1"
  if [[ "$M" == "baseline" ]]; then
    echo "Refusing to train/interpret baseline (no classifier)."; return 0
  fi
  echo ""
  echo "===== MODEL: $M ====="
  echo "[1/3] Training..."
  "$PY" -m src.train     --config configs/default.yaml --model "$M"
  echo "[2/3] Evaluating on TEST..."
  "$PY" -m src.evaluate  --config configs/default.yaml --model "$M"
  echo "[3/3] Interpreting (FP/FN from VAL)..."
  "$PY" -m src.interpret --config configs/default.yaml --model "$M" --k 3
}

case "$MODELS" in
  baseline) run_baseline ;;
  logreg)   run_classifier logreg ;;
  pac)      run_classifier pac ;;
  lsvc)     run_classifier lsvc ;;
  all)      run_baseline; run_classifier logreg; run_classifier pac; run_classifier lsvc ;;
  *)        echo "Unknown arg '$MODELS' (use: baseline | logreg | pac | lsvc | all)"; exit 2 ;;
esac
