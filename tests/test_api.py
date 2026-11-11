import io

from conftest import FakePredictor, png_bytes
from model_api.app import create_api


def make(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    return create_api(FakePredictor()).test_client()


def test_api_requires_key(monkeypatch):
    c = make(monkeypatch)
    r = c.post("/predict", data={"image": (io.BytesIO(png_bytes()), "a.png")}, content_type="multipart/form-data")
    assert r.status_code == 401


def test_api_predicts_with_key(monkeypatch):
    c = make(monkeypatch)
    r = c.post("/predict", data={"image": (io.BytesIO(png_bytes()), "a.png")}, content_type="multipart/form-data",
               headers={"X-API-Key": "secret"})
    assert r.status_code == 200 and r.json["label"] == "AI-generated"
