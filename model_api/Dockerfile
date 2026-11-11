# Model API image (Task 7). Build from repo root: docker build -f model_api/Dockerfile -t visionguard-api .
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0
WORKDIR /app
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt
COPY vg/__init__.py vg/preprocess.py vg/
COPY app/__init__.py app/db.py app/inference.py app/validation.py app/
COPY model_api model_api
# the champion model is baked into the image: the server never calls DagsHub per request
COPY models/champion.keras models/champion_meta.json /models/
ENV MODEL_PATH=/models/champion.keras
RUN useradd --create-home --uid 10001 apiuser
USER apiuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)"
# one worker: each worker would load its own copy of the model into RAM; long timeout for slow free CPUs
CMD ["gunicorn", "-w", "1", "--threads", "4", "--timeout", "300", "-b", "0.0.0.0:8000", "model_api.app:app"]