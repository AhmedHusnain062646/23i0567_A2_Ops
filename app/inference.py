"""Task 4.1 / 8 - inference. Model is loaded ONCE at start-up (not per request).
LocalPredictor runs the Keras model in-process; RemotePredictor calls the cloud-hosted model API."""
import json
import time
from pathlib import Path

IMG_SIZE_DEFAULT = 224


class PredictionError(RuntimeError):
    pass


def _result(p_ai, threshold, name, version, started):
    is_ai = p_ai >= threshold
    return {"label": "AI-generated" if is_ai else "Real",
            "confidence": float(p_ai if is_ai else 1 - p_ai),
            "prob_ai": float(p_ai), "threshold": float(threshold),
            "model_name": name, "model_version": str(version),
            "latency_ms": int((time.perf_counter() - started) * 1000)}


class LocalPredictor:
    source = "local"

    def __init__(self, model_path, threshold=None):
        import tensorflow as tf
        from tensorflow import keras
        from vg.preprocess import decode_resize
        self._tf, self._decode = tf, decode_resize
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found at {path} (check MODEL_PATH / volume mount)")
        meta_file = path.with_name("champion_meta.json")
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        self.model = keras.models.load_model(path, compile=False)
        self.size = int(meta.get("img_size", IMG_SIZE_DEFAULT))
        self.threshold = float(threshold if threshold is not None else meta.get("threshold", 0.5))
        self.name = meta.get("model_name", "VisionGuard")
        self.version = meta.get("model_version", "local")

    def predict(self, image_bytes):
        t0 = time.perf_counter()
        img = self._decode(image_bytes, self.size)  # SAME function as in training
        p = float(self.model(self._tf.expand_dims(img, 0), training=False).numpy()[0, 0])
        return _result(p, self.threshold, self.name, self.version, t0)


class RemotePredictor:
    source = "remote"

    def __init__(self, url, api_key, timeout=30):
        self.url, self.key, self.timeout = url.rstrip("/"), api_key, timeout

    def predict(self, image_bytes):
        import requests
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{self.url}/predict", files={"image": ("upload.png", image_bytes)},
                              headers={"X-API-Key": self.key}, timeout=self.timeout)
        except requests.RequestException as e:
            raise PredictionError(f"Could not reach the model API: {e}") from e
        if r.status_code != 200:
            raise PredictionError(f"Model API returned HTTP {r.status_code}: {r.text[:200]}")
        d = r.json()
        d["latency_ms"] = int((time.perf_counter() - t0) * 1000)  # round-trip time as seen by the web app
        return d


def build_predictor(cfg):
    if cfg["INFERENCE_MODE"] == "remote":
        return RemotePredictor(cfg["REMOTE_API_URL"], cfg["REMOTE_API_KEY"])
    return LocalPredictor(cfg["MODEL_PATH"], cfg["THRESHOLD"])
