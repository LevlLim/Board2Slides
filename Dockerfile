FROM node:22-bookworm-slim AS nodebase

FROM python:3.12-slim-bookworm

ARG WITH_MODELS=true
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/user \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    PADDLE_PDX_CACHE_HOME=/app/.cache/paddlex \
    HF_HOME=/app/.cache/huggingface \
    PATH=/app/.venv-web/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=nodebase /usr/local/bin/node /usr/local/bin/node
COPY --from=nodebase /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && useradd --create-home --uid 1000 user \
    && mkdir -p /app \
    && chown user:user /app

USER user
WORKDIR /app
COPY --chown=user:user requirements/ /app/requirements/
COPY --chown=user:user package.json package-lock.json /app/
RUN python -m venv .venv-web \
    && .venv-web/bin/python -m pip install --no-cache-dir -r requirements/web.txt \
    && npm ci --omit=dev
RUN if [ "$WITH_MODELS" = "true" ]; then \
      python -m venv .venv-ocr \
      && .venv-ocr/bin/python -m pip install --no-cache-dir -r requirements/ocr.txt \
      && python -m venv .venv-llm \
      && .venv-llm/bin/python -m pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 \
      && .venv-llm/bin/python -m pip install --no-cache-dir -r requirements/llm.txt; \
    fi

COPY --chown=user:user scripts/download_base.py /app/scripts/download_base.py
RUN mkdir -p /app/models/llm/base /app/data/jobs /app/.cache/paddlex \
    && if [ "$WITH_MODELS" = "true" ]; then \
      .venv-ocr/bin/python -c "import paddle, paddleocr" \
      && .venv-llm/bin/python -c "import torch, transformers, peft"; \
    fi

COPY --chown=user:user app/ /app/app/
COPY --chown=user:user renderer/ /app/renderer/
COPY --chown=user:user models/ocr/ /app/models/ocr/
COPY --chown=user:user models/llm/adapter/ /app/models/llm/adapter/

EXPOSE 8000
CMD ["sh", "-c", "set -e; if [ -x .venv-llm/bin/python ]; then .venv-llm/bin/python scripts/download_base.py; fi; exec python -m uvicorn app:app --app-dir /app/app --host 0.0.0.0 --port 8000"]
