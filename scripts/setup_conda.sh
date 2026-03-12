#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-diamond_env}"
PYTHON_VERSION="${2:-3.10}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  echo "Detected active virtualenv at '$VIRTUAL_ENV'. Ignoring it for conda setup."
  unset VIRTUAL_ENV
fi

# Prevent inherited Python env vars from leaking into conda operations.
unset PYTHONHOME 2>/dev/null || true
unset PYTHONPATH 2>/dev/null || true

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

ENV_PREFIX="$(conda env list | awk -v env="$ENV_NAME" '$1 == env {print $NF}')"
if [[ -z "$ENV_PREFIX" ]]; then
  echo "Error: could not resolve prefix for conda environment '$ENV_NAME'."
  exit 1
fi

PY_CMD="$ENV_PREFIX/bin/python"
if [[ ! -x "$PY_CMD" ]]; then
  echo "Error: python executable not found at '$PY_CMD'."
  exit 1
fi

# Keep installs and checks isolated from user-level site-packages.
export PYTHONNOUSERSITE=1

echo "Verifying interpreter from environment '$ENV_NAME'..."
PY_EXE="$("$PY_CMD" -c 'import sys; print(sys.executable)')"
PY_VER="$("$PY_CMD" -c 'import sys; print(sys.version)')"
echo "Python executable: $PY_EXE"
echo "Python version: $PY_VER"

if [[ "$PY_EXE" == *".venv"* ]]; then
  echo "Error: interpreter points to .venv. Environment isolation failed."
  exit 1
fi

echo "Installing base requirements from requirements.txt..."
"$PY_CMD" -m pip install -r "$ROOT_DIR/requirements.txt"

echo "Verifying TensorFlow import..."
"$PY_CMD" -c 'import tensorflow as tf; print("TensorFlow", tf.__version__)'

echo "Setup complete. Use this environment with: conda activate $ENV_NAME"
