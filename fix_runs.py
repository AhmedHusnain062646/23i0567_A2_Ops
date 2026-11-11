import os
from dotenv import load_dotenv
import mlflow
from mlflow import MlflowClient

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
print("mlflow", mlflow.__version__)
c = MlflowClient()

RUNS = {
    "mobilenet_ftlr1e4": "e872ccb1a6ef472595f2502a11572251",
    "mobilenet_drop04":  "f0f853f30e174bec85ded015998057ac",
    "mobilenet_auto":    "2a449488a8144ee4ad87d1972298cd09",
    "mobilenet_manual":  "6995d7e0f0ca4b0ba66b5c9ca27cb5e1",
}
for name, rid in RUNS.items():
    if c.get_run(rid).info.status == "RUNNING":
        c.set_terminated(rid, "FINISHED")
    arts = [a.path for a in c.list_artifacts(rid)]
    print(name, c.get_run(rid).info.status, arts)

exp = c.get_experiment_by_name("VisionGuard")
try:
    for m in mlflow.search_logged_models(experiment_ids=[exp.experiment_id], output_format="list"):
        print("LOGGED MODEL:", m.name, m.model_id, "run:", m.source_run_id)
except Exception as e:
    print("logged-model search not available:", e)