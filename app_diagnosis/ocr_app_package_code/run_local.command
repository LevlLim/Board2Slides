#!/bin/bash
# Local (Mac/Linux) launch of the packaged app on http://127.0.0.1:8501 using the package's own environment.
set -euo pipefail
PKG="$(cd -- "$(dirname -- "$0")" && pwd)"
PY="${B2N_PYTHON:-$PKG/.venv/bin/python}"
cd "$PKG/b2n_ocr"
exec "$PY" -m streamlit run app.py --server.address 127.0.0.1 --server.port "${PORT:-8501}" \
  --server.headless true --browser.gatherUsageStats false --server.maxUploadSize 30
