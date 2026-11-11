# VisionGuard web app (Task 6.2)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Dependencies first so a code-only change reuses the TensorFlow layer.
COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

COPY vg/__init__.py vg/preprocess.py vg/
COPY app app

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:5000/health', timeout=4).status==200 else 1)"

CMD ["python", "-m", "app"]
