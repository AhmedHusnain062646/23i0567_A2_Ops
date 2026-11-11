import os, mlflow, pandas as pd
from mlflow.tracking import MlflowClient

for line in open(".env", encoding="utf-8"):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
c = MlflowClient()
runs = mlflow.search_runs(experiment_ids=["1"])
for _, r in runs.iterrows():
    rid = r["run_id"]
    print(r["tags.mlflow.runName"], rid[:8])
    for a in c.list_artifacts(rid):
        sub = [x.path.split("/")[-1] for x in c.list_artifacts(rid, a.path)] if a.is_dir else ""
        print("   ", a.path, sub)
    tags = {k: v for k, v in r.items() if k.startswith("tags.") and "mlflow." not in k and pd.notna(v)}
    print("    tags:", tags)