import os, mlflow
from mlflow.tracking import MlflowClient

# load .env without printing any values
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    for line in open(".env", encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

print("URI set:", bool(os.environ.get("MLFLOW_TRACKING_URI")))
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
c = MlflowClient()

runs = mlflow.search_runs(search_all_experiments=True)
print(len(runs), "runs")
cols = [x for x in ("tags.mlflow.runName", "status", "experiment_id") if x in runs.columns]
print(runs[cols].to_string())

for m in c.search_registered_models():
    print("MODEL", m.name)
    for v in c.search_model_versions(f"name='{m.name}'"):
        print("  v", v.version, "aliases:", v.aliases, "run:", v.run_id[:8])

latest = runs.sort_values("start_time", ascending=False).head(3)["run_id"]
for rid in latest:
    print("ARTIFACTS for", rid[:8])
    for a in c.list_artifacts(rid):
        print("  -", a.path)