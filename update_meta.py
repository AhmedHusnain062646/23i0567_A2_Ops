# update_meta.py
import os, json, pathlib
from dotenv import load_dotenv
import mlflow
from mlflow import MlflowClient

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
c = MlflowClient()
mv = c.get_model_version_by_alias("VisionGuard", "champion")
m = c.get_run(mv.run_id).data.metrics
print("champion version", mv.version, "run", mv.run_id)
for k in ("test_threshold", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"):
    print(f"  {k:15} {m.get(k)}")

p = pathlib.Path("models/champion_meta.json")
meta = json.loads(p.read_text(encoding="utf-8"))
if "test_threshold" in m:
    meta["threshold"] = float(m["test_threshold"])
    meta["model_version"] = str(mv.version)
    p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("meta updated, threshold =", meta["threshold"])
else:
    print("no test_threshold on the run, paste this output")