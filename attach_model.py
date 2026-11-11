import os, mlflow
for line in open(".env", encoding="utf-8"):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

MODELS = {  # run name -> model file on disk (edit to match your models/ folder)
    "baseline_manual-manual":  "models/baseline.keras",
    "mobilenet_manual-manual": "models/mobilenet.keras",
    "mobilenet_drop04-manual": "models/mobilenet_drop04.keras",
    "mobilenet_img160-manual": "models/mobilenet_img160.keras",
    "mobilenet_lr5e4-manual":  "models/mobilenet_lr5e4.keras",
}
runs = mlflow.search_runs(experiment_ids=["1"])
for name, path in MODELS.items():
    m = runs[(runs["tags.mlflow.runName"] == name) & runs["tags.seed"].notna()]
    if m.empty or not os.path.exists(path):
        print("SKIP", name, path); continue
    rid = m.iloc[0]["run_id"]
    with mlflow.start_run(run_id=rid):
        mlflow.log_artifact(path, artifact_path="model")
    print("attached", name, rid[:8])