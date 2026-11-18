#!/bin/sh
# Container / Linux server start. Streamlit reads b2n_ocr/.streamlit/config.toml when started from b2n_ocr/.
set -eu
cd "$(dirname "$0")/../b2n_ocr"
exec python -m streamlit run app.py \
  --server.address "${HOST:-0.0.0.0}" --server.port "${PORT:-8501}" \
  --server.headless true --browser.gatherUsageStats false --server.maxUploadSize 30
