# Minimal CPU image for the Hugging Face Space (no CUDA, no training deps).
FROM python:3.11-slim

# CPU-only torch — keeps the image small (no multi-GB CUDA wheels)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# non-root user (HF Spaces convention) with a writable home for the HF model cache
RUN useradd -m -u 1000 user
COPY --chown=user . /app
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    HF_HUB_OFFLINE=0 \
    SQLFORGE_BASE=Qwen/Qwen2.5-Coder-1.5B-Instruct \
    SQLFORGE_ADAPTER=Abdullahkousa2/sqlforge-qwen2.5-coder-1.5b \
    SQLFORGE_4BIT=0 \
    SQLFORGE_DB_DIR=/app/demo_dbs

EXPOSE 8000
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
