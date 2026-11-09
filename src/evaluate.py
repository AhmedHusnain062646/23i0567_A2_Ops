# src/evaluate.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, precision_recall_curve)
import data
from data import make_ds


def evaluate(run, model):
    """Threshold from validation (max F1), one pass over test. Returns (metrics dict, figure paths)."""
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    data.IMG_SIZE = model.input_shape[1]

    def predict(split):
        p = model.predict(make_ds(split, training=False), verbose=0).ravel()
        return pd.read_csv(f"data/splits/{split}.csv"), p

    vdf, vp = predict("val")
    prec, rec, thr = precision_recall_curve(vdf["label"], vp)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    t = float(thr[np.argmax(f1s)])

    tdf, tp = predict("test")
    y = tdf["label"].values
    pred = (tp >= t).astype(int)
    row = {"model": run, "threshold": round(t, 3),
           "accuracy": accuracy_score(y, pred),
           "precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred),
           "f1": f1_score(y, pred),
           "roc_auc": roc_auc_score(y, tp)}

    cm_path = f"reports/figures/{run}_confusion.png"
    ConfusionMatrixDisplay(confusion_matrix(y, pred), display_labels=["Real", "AI"]).plot(cmap="Blues")
    plt.title(f"{run}: confusion matrix (test, threshold {t:.2f})")
    plt.savefig(cm_path, dpi=150); plt.close()

    roc_path = f"reports/figures/{run}_roc.png"
    fpr, tpr, _ = roc_curve(y, tp)
    plt.plot(fpr, tpr, label=f"AUC = {row['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate")
    plt.ylabel("True positive rate"); plt.title(f"{run}: ROC (test)"); plt.legend()
    plt.savefig(roc_path, dpi=150); plt.close()

    # 2.6: 10 most confidently wrong test images
    tdf["prob_ai"] = tp
    tdf["pred"] = pred
    wrong = tdf[tdf["label"] != tdf["pred"]].copy()
    wrong["conf"] = np.abs(wrong["prob_ai"] - 0.5)
    wrong = wrong.sort_values("conf", ascending=False).head(10)
    wrong.to_csv(f"reports/{run}_misclassified.csv", index=False)
    mis_path = f"reports/figures/{run}_misclassified.png"
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    for ax, (_, r) in zip(axes.ravel(), wrong.iterrows()):
        ax.imshow(tf.keras.utils.load_img(r["path"]))
        ax.set_title(f"true={'AI' if r['label'] else 'Real'}  p(AI)={r['prob_ai']:.2f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{run}: 10 misclassified test images")
    fig.tight_layout(); fig.savefig(mis_path, dpi=150); plt.close(fig)
    return row, [cm_path, roc_path, mis_path]


if __name__ == "__main__":
    rows = []
    for name in ["baseline", "mobilenet"]:
        m = tf.keras.models.load_model(f"models/{name}.keras")
        row, _ = evaluate(name, m)
        rows.append(row)
    table = pd.DataFrame(rows).round(4)
    table.to_csv("reports/test_metrics.csv", index=False)
    print(table.to_string(index=False))