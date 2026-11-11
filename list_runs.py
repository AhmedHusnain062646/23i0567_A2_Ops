import os
from dotenv import load_dotenv
import mlflow
from mlflow import MlflowClient

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
c = MlflowClient()
exp = c.get_experiment_by_name("VisionGuard")
nan = float("nan")
for r in c.search_runs(exp.experiment_id, order_by=["attributes.start_time DESC"]):
    arts = [a.path for a in c.list_artifacts(r.info.run_id)]
    m, t = r.data.metrics, r.data.tags
    print(f"{r.info.run_name:30} {r.info.run_id} {r.info.status:9} "
          f"test_auc={m.get('test_roc_auc', nan):.4f} val_loss={m.get('final_val_loss_best', nan):.4f} "
          f"model={'model' in arts or 'model_file' in arts} "
          f"tags={'git_commit' in t} dirty={t.get('git_dirty')}")