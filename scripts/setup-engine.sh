#!/usr/bin/env bash
# Prepare the local beauty engine: a private venv with MediaPipe/OpenCV and the
# FaceLandmarker model. Safe to re-run; nothing is installed into system Python.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
MODEL_DIR="$ROOT/engine/models"
MODEL="$MODEL_DIR/face_landmarker.task"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
SEG_MODEL="$MODEL_DIR/selfie_multiclass_256x256.tflite"
SEG_URL="https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"

pick_python() {
  for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        echo "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

PYTHON="$(pick_python || true)"
if [ -z "${PYTHON}" ]; then
  echo "error: need Python 3.10+ (python3.12 recommended)" >&2
  exit 1
fi
echo "==> Python: $PYTHON ($($PYTHON -V 2>&1))"

if [ ! -x "$VENV/bin/python" ]; then
  echo "==> Creating venv at $VENV"
  "$PYTHON" -m venv "$VENV"
fi

echo "==> Installing engine dependencies"
"$VENV/bin/python" -m pip install --upgrade pip >/dev/null
"$VENV/bin/python" -m pip install -r "$ROOT/engine/requirements.txt"

mkdir -p "$MODEL_DIR"
if [ ! -s "$MODEL" ]; then
  echo "==> Downloading FaceLandmarker model (3.6 MB)"
  curl -fsSL "$MODEL_URL" -o "$MODEL.tmp"
  mv "$MODEL.tmp" "$MODEL"
fi
if [ ! -s "$SEG_MODEL" ]; then
  echo "==> Downloading selfie segmentation model (16 MB)"
  curl -fsSL "$SEG_URL" -o "$SEG_MODEL.tmp"
  mv "$SEG_MODEL.tmp" "$SEG_MODEL"
fi

echo "==> Smoke test"
"$VENV/bin/python" - <<'PY'
import cv2, numpy, mediapipe
from mediapipe.tasks.python import vision
print("engine ready:", cv2.__version__, numpy.__version__, mediapipe.__version__)
PY
echo "==> Done. Restart DSH to load the plugin."
