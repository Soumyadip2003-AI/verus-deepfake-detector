#!/bin/sh
set -eu
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ] || [ ! -f backend/models/gend-clip-l14.safetensors ]; then
  echo "Setup is incomplete. Follow the Run locally steps in README.md first."
  exit 1
fi

if curl -fsS --max-time 2 http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
  echo "Detector is already running at http://127.0.0.1:8000"
  open http://127.0.0.1:8000
  exit 0
fi

echo "Open http://127.0.0.1:8000 in your browser. The result appears below your uploaded media."
exec .venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
