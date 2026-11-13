FROM node:22-bookworm-slim AS nodebase

FROM python:3.13-slim
COPY --from=nodebase /usr/local/bin/node /usr/local/bin/node
COPY --from=nodebase /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm

WORKDIR /app
COPY requirements /app/requirements
COPY package.json package-lock.json /app/
RUN pip install --no-cache-dir -r requirements/web.txt \
    && npm ci --omit=dev

# Model-enabled CPU image is optional; mount the 4.1 GB Qwen base at runtime.
ARG WITH_MODELS=false
RUN if [ "$WITH_MODELS" = "true" ]; then \
      apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
      && rm -rf /var/lib/apt/lists/* \
      && pip install --no-cache-dir -r requirements/ocr.txt \
      && pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
      && pip install --no-cache-dir -r requirements/llm.txt; \
    fi

COPY app /app/app
COPY renderer /app/renderer
COPY models/ocr /app/models/ocr
COPY models/llm/adapter /app/models/llm/adapter
RUN mkdir -p /app/models/llm/base /app/data

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app:app", "--app-dir", "/app/app", "--host", "0.0.0.0", "--port", "8000"]
