# Board2Notes OCR - Streamlit student app with packaged weights (no downloads at runtime). linux/amd64.
# Build from the package folder:   docker build --platform linux/amd64 -f deploy/Dockerfile -t board2notes-ocr:2026-11-09 .

# Stage 1: normalise permissions. Package files may be owner-only (600/700) on the host; COPY keeps those modes
# and makes root the owner, so the non-root runtime user could not read them. Fixing them here, then copying the
# result, avoids duplicating the 86 MB of weights in an extra chmod layer.
FROM python:3.12-slim-bookworm AS source
COPY . /src
RUN chmod -R u=rwX,go=rX /src && chmod 0755 /src/deploy/start.sh

FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True PADDLE_PDX_CACHE_HOME=/tmp/paddlex \
    B2N_PACKAGE_ROOT=/app B2N_OCR_MODEL=S1 HOME=/tmp
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=source /src/requirements/ requirements/
RUN pip install -r requirements/base.txt \
 && pip install --no-deps --force-reinstall -r requirements/opencv-override.txt
COPY --from=source /src/ /app/
# fail the build if any model/code/fixture file is missing, altered or unexpected
RUN python tools/verify_package.py
# app files stay root-owned and read-only for the runtime user; only outputs/ (results, recorded references) is writable
RUN useradd --create-home --uid 10001 b2n && mkdir -p /app/outputs && chown -R 10001 /app/outputs
USER 10001
# build-time proof that the runtime user can read every package file and traverse every folder
RUN python -c "import os,sys; bad=[os.path.join(r,n) for r,ds,fs in os.walk('/app') for n in fs+ds if not os.access(os.path.join(r,n), os.R_OK|(os.X_OK if n in ds else 0))]; print('unreadable:', bad[:5]) if bad else None; sys.exit(1 if bad else 0)" \
 && python tools/verify_package.py
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen('http://127.0.0.1:%s/_stcore/health' % os.environ.get('PORT','8501'), timeout=4)"
CMD ["sh", "deploy/start.sh"]
