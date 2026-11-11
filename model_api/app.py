"""Task 7 - model-serving API, deployed on the cloud VM. Protected by a shared secret (X-API-Key).
POST /predict  (multipart field 'image') -> JSON      GET /health -> JSON"""
import hmac
import os

from flask import Flask, jsonify, request

from app.inference import LocalPredictor
from app.validation import UploadError, validate_upload


def create_api(predictor=None):
    api = Flask(__name__)
    api.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024
    key = os.getenv("API_KEY", "")
    thr = os.getenv("THRESHOLD")
    pred = predictor or LocalPredictor(os.getenv("MODEL_PATH", "/models/champion.keras"),
                                       float(thr) if thr else None)

    @api.get("/health")
    def health():
        return jsonify(status="ok", model=pred.name, version=pred.version)

    @api.post("/predict")
    def predict():
        sent = request.headers.get("X-API-Key", "")
        if not key or not hmac.compare_digest(sent, key):
            return jsonify(error="invalid or missing API key"), 401
        try:
            _, data = validate_upload(request.files.get("image"))
        except UploadError as e:
            return jsonify(error=str(e)), 400
        return jsonify(pred.predict(data))

    @api.errorhandler(413)
    def too_big(e):
        return jsonify(error="file too large"), 413

    return api


app = None
if os.getenv("API_KEY") is not None:  # created lazily for gunicorn: `gunicorn model_api.app:app`
    app = create_api()
