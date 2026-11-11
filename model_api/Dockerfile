# Model API image (Task 7). Build from repo root: docker build -f model_api/Dockerfile -t visionguard-api .
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TF_CPP_MIN_LOG_LEVEL=2
WORKDIR /app
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt
COPY vg/__init__.py vg/preprocess.py vg/
COPY app/__init__.py app/db.py app/inference.py app/validation.py app/
COPY model_api model_api
RUN useradd --create-home --uid 10001 apiuser
USER apiuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)"
# one worker: each worker would load its own copy of the model into RAM (free VMs are small)
CMD ["gunicorn", "-w", "1", "--threads", "4", "--timeout", "120", "-b", "0.0.0.0:8000", "model_api.app:app"]
