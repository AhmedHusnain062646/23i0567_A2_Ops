# src/load_champion.py
"""Task 3.6: Load the 'champion' model from MLflow Model Registry and classify one image.

Usage:
    python src/load_champion.py <path_to_image>

The script connects to DagsHub MLflow, loads the latest 'champion' version of the
VisionGuard model, preprocesses the image (same pipeline as training), and prints
the verdict and confidence.
"""
import os, sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import mlflow
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "VisionGuard"
ALIAS = "champion"
THRESHOLD = float(os.environ.get("VISIONGUARD_THRESHOLD", "0.36"))

def load_model():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    model_uri = f"models:/{MODEL_NAME}@{ALIAS}"
    print(f"Loading model from registry: {model_uri}")
    model = mlflow.keras.load_model(model_uri)
    print(f"Model loaded: input shape = {model.input_shape}")
    return model

def preprocess(img_path: str, img_size: int) -> np.ndarray:
    """Replicates the training preprocessing exactly (resize → float32, model handles rescaling)."""
    raw = tf.io.read_file(img_path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size), method="bilinear", antialias=True)
    img = tf.cast(img, tf.float32)
    return tf.expand_dims(img, 0).numpy()   # batch dim

def classify(model, img_path: str, threshold: float = THRESHOLD):
    img_size = model.input_shape[1]
    x = preprocess(img_path, img_size)
    prob = float(model.predict(x, verbose=0).ravel()[0])
    label = "AI-generated" if prob >= threshold else "Real"
    print(f"\nImage : {img_path}")
    print(f"p(AI) : {prob:.4f}  (threshold = {threshold})")
    print(f"Verdict: {label}")
    return label, prob

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/load_champion.py <path_to_image>")
        sys.exit(1)
    img_path = sys.argv[1]
    if not Path(img_path).is_file():
        print(f"Error: file not found: {img_path}")
        sys.exit(1)
    model = load_model()
    classify(model, img_path)
