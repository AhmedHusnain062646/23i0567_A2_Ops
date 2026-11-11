import io

import pytest
from PIL import Image

from app import create_app


class FakePredictor:
    name = "fake"
    version = "1"
    source = "local"

    def predict(self, data):
        return {"label": "AI-generated", "confidence": 0.91, "prob_ai": 0.91, "threshold": 0.5,
                "model_name": "fake", "model_version": "1", "latency_ms": 3}


@pytest.fixture
def client(tmp_path):
    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path/'t.db'}",
                      "MAX_CONTENT_LENGTH": 200 * 1024}, predictor=FakePredictor())
    return app.test_client()


def png_bytes(size=(32, 32)):
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 30, 200)).save(buf, "PNG")
    return buf.getvalue()
