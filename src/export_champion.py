"""Download the registry alias 'champion' once so the website never calls DagsHub per request.

Usage:
    python src/export_champion.py

Writes models/champion.keras and models/champion_meta.json.
"""
import json
import os
import shutil
import tempfile
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient

load_dotenv()

MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "VisionGuard")
ALIAS = "champion"
DEST = Path("models")


def main():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    print(f"Champion is {MODEL_NAME} version {mv.version} (run {mv.run_id})")
    tmp = tempfile.mkdtemp()
    local = mlflow.artifacts.download_artifacts(f"runs:/{mv.run_id}/model", dst_path=tmp)
    keras_files = list(Path(local).rglob("*.keras")) or list(Path(local).rglob("*.h5"))
    if not keras_files:
        file_dir = mlflow.artifacts.download_artifacts(f"runs:/{mv.run_id}/model_file", dst_path=tmp)
        keras_files = list(Path(file_dir).rglob("*.keras")) or list(Path(file_dir).rglob("*.h5"))
    if not keras_files:
        raise SystemExit(f"No .keras file found for run {mv.run_id}")
    DEST.mkdir(exist_ok=True)
    shutil.copy(keras_files[0], DEST / "champion.keras")
    meta = {
        "img_size": 128,
        "threshold": 0.36,
        "model_name": MODEL_NAME,
        "model_version": str(mv.version),
        "run_id": mv.run_id,
    }
    try:
        run = client.get_run(mv.run_id)
        if "img_size" in run.data.params:
            meta["img_size"] = int(float(run.data.params["img_size"]))
        if "test_threshold" in run.data.metrics:
            meta["threshold"] = float(run.data.metrics["test_threshold"])
        elif "threshold" in run.data.params:
            meta["threshold"] = float(run.data.params["threshold"])
    except Exception as exc:
        print("Could not read run params for meta, using defaults:", exc)
    (DEST / "champion_meta.json").write_text(json.dumps(meta, indent=2))
    print("Wrote", DEST / "champion.keras")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
