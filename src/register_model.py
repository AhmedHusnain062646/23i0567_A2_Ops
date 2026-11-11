# src/register_model.py
"""Task 3.6: Register the best VisionGuard run in MLflow Model Registry.

Usage:
    python src/register_model.py [--run-id <run_id>] [--alias champion]

If --run-id is not given, automatically picks the FINISHED run with the highest
test_roc_auc from the VisionGuard experiment.
"""
import argparse, os
import mlflow
from mlflow import MlflowClient
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "VisionGuard"

ap = argparse.ArgumentParser()
ap.add_argument("--run-id", default=None, help="MLflow run ID to register (auto-picks best if omitted)")
ap.add_argument("--alias", default="champion", help="Alias to give the registered version")
args = ap.parse_args()

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = MlflowClient()

# Find the experiment
exp = client.get_experiment_by_name("VisionGuard")
if exp is None:
    raise RuntimeError("VisionGuard experiment not found on DagsHub")

# Pick best run by test_roc_auc
if args.run_id:
    run_id = args.run_id
    run = client.get_run(run_id)
    print(f"Using specified run: {run.info.run_name} ({run_id})")
else:
    runs = client.search_runs(
        exp.experiment_id,
        filter_string="status = 'FINISHED'",
        order_by=["metrics.test_roc_auc DESC"],
        max_results=10,
    )
    if not runs:
        raise RuntimeError("No finished runs with test_roc_auc found")
    best = runs[0]
    run_id = best.info.run_id
    print(f"Best run: {best.info.run_name} ({run_id})")
    print(f"  test_roc_auc = {best.data.metrics.get('test_roc_auc', 'N/A'):.4f}")
    print(f"  test_f1      = {best.data.metrics.get('test_f1', 'N/A'):.4f}")

# Try to register the Keras model artifact from the run
# The model artifact is logged under "model/" by mlflow.keras.log_model
model_uri = f"runs:/{run_id}/model"
print(f"\nRegistering model from: {model_uri}")

try:
    result = mlflow.register_model(model_uri, MODEL_NAME)
    version = result.version
    print(f"Registered as '{MODEL_NAME}' version {version}")
except Exception as e:
    print(f"Could not register from 'model/' artifact: {e}")
    # Try model_file fallback
    model_uri = f"runs:/{run_id}/model_file"
    print(f"Trying fallback: {model_uri}")
    result = mlflow.register_model(model_uri, MODEL_NAME)
    version = result.version
    print(f"Registered (fallback) as '{MODEL_NAME}' version {version}")

# Set description and alias
client.update_registered_model(
    name=MODEL_NAME,
    description="VisionGuard: MobileNetV2 transfer learning for AI vs Real image classification."
)
client.update_model_version(
    name=MODEL_NAME,
    version=version,
    description=f"Trained from run {run_id}. Best by test_roc_auc."
)
client.set_registered_model_alias(MODEL_NAME, args.alias, version)
print(f"\nAlias '{args.alias}' → version {version}")
print("\nTo load this model:")
print(f"  mlflow.keras.load_model('models:/{MODEL_NAME}@{args.alias}')")
