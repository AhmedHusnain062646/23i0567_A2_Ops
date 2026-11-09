# src/evaluate.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, precision_recall_curve)
import data
from data import make_ds

Path("reports/figures").mkdir(parents=True, exist_ok=True)
rows = []

for name in ["baseline", "mobilenet"]:
    model = tf.keras.models.load_model(f"models/{name}.keras")
    data.IMG_SIZE = model.input_shape[1]          # use the size this model was trained at
    print(f"{name}: input size {data.IMG_SIZE}")

    def predict(split):
        p = model.predict(make_ds(split, training=False), verbose=0).ravel()
        df = pd.read_csv(f"data/splits/{split}.csv")
        return df, p

    # ---- choose threshold on VALIDATION data only ----
    vdf, vp = predict("val")
    prec, rec, thr = precision_recall_curve(vdf["label"], vp)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    t = float(thr[np.argmax(f1s)])
    print(f"{name}: threshold from validation (max F1) = {t:.3f}")

    # ---- test set, evaluated once ----
    tdf, tp = predict("test")
    y = tdf["label"].values
    pred = (tp >= t).astype(int)
    rows.append({"model": name, "threshold": round(t, 3),
                 "accuracy": accuracy_score(y, pred),
                 "precision": precision_score(y, pred, zero_division=0),
                 "recall": recall_score(y, pred),
                 "f1": f1_score(y, pred),
                 "roc_auc": roc_auc_score(y, tp)})

    cm = confusion_matrix(y, pred)
    ConfusionMatrixDisplay(cm, display_labels=["Real", "AI"]).plot(cmap="Blues")
    plt.title(f"{name}: confusion matrix (test, threshold {t:.2f})")
    plt.savefig(f"reports/figures/{name}_confusion.png", dpi=150); plt.close()

    fpr, tpr, _ = roc_curve(y, tp)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y, tp):.3f}")
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate")
    plt.ylabel("True positive rate"); plt.title(f"{name}: ROC (test)"); plt.legend()
    plt.savefig(f"reports/figures/{name}_roc.png", dpi=150); plt.close()

    # ---- 2.6: 10 most confidently wrong test images ----
    tdf["prob_ai"] = tp
    tdf["pred"] = pred
    wrong = tdf[tdf["label"] != tdf["pred"]].copy()
    wrong["conf"] = np.abs(wrong["prob_ai"] - 0.5)
    wrong = wrong.sort_values("conf", ascending=False).head(10)
    wrong.to_csv(f"reports/{name}_misclassified.csv", index=False)
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    for ax, (_, r) in zip(axes.ravel(), wrong.iterrows()):
        ax.imshow(tf.keras.utils.load_img(r["path"]))
        ax.set_title(f"true={'AI' if r['label'] else 'Real'}  p(AI)={r['prob_ai']:.2f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{name}: 10 misclassified test images")
    fig.tight_layout(); fig.savefig(f"reports/figures/{name}_misclassified.png", dpi=150)
    plt.close(fig)

table = pd.DataFrame(rows).round(4)
table.to_csv("reports/test_metrics.csv", index=False)
print(table.to_string(index=False))