"""Task 3.5 - comparison table and chart from the VisionGuard MLflow experiment.

Usage (from the project root, with .env loaded):
    python src/compare_runs.py
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from dotenv import load_dotenv

load_dotenv()

COLS = [
    "tags.mlflow.runName",
    "params.model",
    "params.img_size",
    "params.lr_stage1",
    "params.dropout",
    "params.seed",
    "metrics.final_val_loss_best",
    "metrics.test_roc_auc",
    "metrics.test_f1",
    "metrics.test_recall",
    "metrics.test_precision",
    "metrics.test_accuracy",
]


def main():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    exp = mlflow.get_experiment_by_name(os.getenv("MLFLOW_EXPERIMENT_NAME", "VisionGuard"))
    if exp is None:
        raise SystemExit("VisionGuard experiment not found. Check MLFLOW_TRACKING_URI in .env.")
    df = mlflow.search_runs([exp.experiment_id], order_by=["metrics.test_roc_auc DESC"])
    keep = [c for c in COLS if c in df.columns]
    table = df[keep].copy()
    num = table.select_dtypes(include="number").columns
    table[num] = table[num].round(4)
    out = Path("reports")
    out.mkdir(exist_ok=True)
    csv_path = out / "run_comparison.csv"
    table.to_csv(csv_path, index=False)
    print(table.to_string(index=False))
    print(f"\nWrote {csv_path} ({len(table)} runs)")

    name_col = "tags.mlflow.runName"
    metric_cols = [c for c in ("metrics.test_roc_auc", "metrics.test_f1") if c in table.columns]
    if name_col in table.columns and metric_cols:
        plot_df = table.dropna(subset=metric_cols).set_index(name_col)[metric_cols]
        if not plot_df.empty:
            ax = plot_df.plot.barh(figsize=(9, max(4, 0.45 * len(plot_df))))
            ax.set_title("VisionGuard runs (test ROC-AUC and F1)")
            ax.set_xlabel("score")
            plt.tight_layout()
            png = out / "run_comparison.png"
            plt.savefig(png, dpi=130)
            print(f"Wrote {png}")


if __name__ == "__main__":
    main()
