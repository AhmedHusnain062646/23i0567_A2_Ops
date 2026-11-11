"""Tasks 4 & 5 - Flask web app. All settings come from environment variables (Task 4.4)."""
import base64
import io
import os

from flask import Flask, jsonify, render_template, request
from PIL import Image
from sqlalchemy import text

from .db import Prediction, db
from .inference import PredictionError, build_predictor
from .validation import UploadError, validate_upload


def _env_float(name):
    v = os.getenv(name)
    return float(v) if v not in (None, "") else None


def create_app(test_config=None, predictor=None):
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.getenv("SECRET_KEY", "dev-only-change-me"),
        SQLALCHEMY_DATABASE_URI=os.getenv("DATABASE_URL", "sqlite:///visionguard.db"),
        MAX_CONTENT_LENGTH=int(os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024,
        MODEL_PATH=os.getenv("MODEL_PATH", "models/champion.keras"),
        THRESHOLD=_env_float("THRESHOLD"),
        INFERENCE_MODE=os.getenv("INFERENCE_MODE", "local"),
        REMOTE_API_URL=os.getenv("REMOTE_API_URL", ""),
        REMOTE_API_KEY=os.getenv("REMOTE_API_KEY", ""),
    )
    if test_config:
        app.config.update(test_config)
    db.init_app(app)
    with app.app_context():
        db.create_all()
    app.predictor = predictor or build_predictor(app.config)  # loaded once, reused by every request

    @app.get("/")
    def index():
        return render_template("index.html", max_mb=app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024))

    @app.post("/predict")
    def predict():
        name, data = validate_upload(request.files.get("image"))
        res = app.predictor.predict(data)
        row = Prediction(filename=name[:255], label=res["label"], confidence=res["confidence"],
                         prob_ai=res["prob_ai"], threshold=res["threshold"], model_name=res["model_name"],
                         model_version=res["model_version"], source=app.predictor.source,
                         latency_ms=res.get("latency_ms"))
        db.session.add(row)
        db.session.commit()
        with Image.open(io.BytesIO(data)) as im:  # in-memory preview, never saved to disk
            im = im.convert("RGB")
            im.thumbnail((420, 420))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=85)
        preview = base64.b64encode(buf.getvalue()).decode()
        return render_template("result.html", r=res, filename=name, preview=preview, row=row)

    @app.get("/history")
    def history():
        rows = Prediction.query.order_by(Prediction.created_at.desc()).limit(200).all()
        total = Prediction.query.count()
        return render_template("history.html", rows=rows, total=total)

    @app.get("/health")
    def health():
        try:
            db.session.execute(text("SELECT 1"))
            db_ok = True
        except Exception:  # noqa: BLE001
            db_ok = False
        body = {"status": "ok" if db_ok else "degraded", "database": db_ok,
                "inference_mode": app.config["INFERENCE_MODE"], "model_loaded": app.predictor is not None}
        return jsonify(body), (200 if db_ok else 503)

    def _err(code, title, msg):
        return render_template("error.html", code=code, title=title, msg=msg), code

    @app.errorhandler(UploadError)
    def _bad_upload(e):
        return _err(400, "Upload problem", str(e))

    @app.errorhandler(413)
    def _too_big(e):
        mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        return _err(413, "File too large", f"Please upload an image smaller than {mb} MB.")

    @app.errorhandler(PredictionError)
    def _pred_fail(e):
        app.logger.error("prediction failed: %s", e)
        return _err(502, "Model unavailable", "The model service could not process your image. Please try later.")

    @app.errorhandler(404)
    def _nf(e):
        return _err(404, "Page not found", "That page does not exist.")

    @app.errorhandler(500)
    def _ise(e):
        return _err(500, "Something went wrong", "An unexpected error occurred. It has been logged.")

    return app
