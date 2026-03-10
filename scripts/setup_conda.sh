#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-diamond_env}"
PYTHON_VERSION="${2:-3.10}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda is not installed or not on PATH."
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
else
  echo "Conda environment '$ENV_NAME' already exists."
fi

echo "Verifying interpreter from environment '$ENV_NAME'..."
PY_EXE="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"
PY_VER="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.version)')"
echo "Python executable: $PY_EXE"
echo "Python version: $PY_VER"

if [[ "$PY_EXE" == *".venv"* ]]; then
  echo "Error: interpreter points to .venv. Environment isolation failed."
  exit 1
fi

echo "Installing base requirements from requirements.txt..."
conda run -n "$ENV_NAME" python -m pip install -r "$ROOT_DIR/requirements.txt"

echo "Verifying TensorFlow import..."
conda run -n "$ENV_NAME" python -c 'import tensorflow as tf; print("TensorFlow", tf.__version__)'

echo "Setup complete. Use this environment with: conda activate $ENV_NAME"
