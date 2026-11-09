# src/train_baseline.py
import json, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from data import make_ds
from models import build_baseline

SEED, EPOCHS, LR = 42, 40, 1e-3
tf.keras.utils.set_random_seed(SEED)          # seeds python, numpy and TF together
os.environ["PYTHONHASHSEED"] = str(SEED)

Path("models").mkdir(exist_ok=True); Path("reports/figures").mkdir(parents=True, exist_ok=True)
train_ds = make_ds("train", training=True, seed=SEED)
val_ds = make_ds("val", training=False)

model = build_baseline()
model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")],
)
model.summary()

# per-layer table for the report
rows = [(l.name, str(l.output.shape), l.count_params()) for l in model.layers]
pd.DataFrame(rows, columns=["layer", "output_shape", "params"]).to_csv(
    "reports/baseline_layers.csv", index=False)

cbs = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                     restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("models/baseline.keras", monitor="val_loss",
                                       save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                         patience=3, min_lr=1e-6, verbose=1),
]
hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cbs)

h = hist.history
best = int(np.argmin(h["val_loss"]))          # 0-based
print(f"Best epoch: {best + 1}  val_loss={h['val_loss'][best]:.4f} "
      f"val_acc={h['val_accuracy'][best]:.4f}")
json.dump({k: [float(v) for v in vals] for k, vals in h.items()},
          open("reports/baseline_history.json", "w"))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for a, (m, title) in zip(ax, [("loss", "Loss"), ("accuracy", "Accuracy")]):
    a.plot(h[m], label="train"); a.plot(h[f"val_{m}"], label="validation")
    a.axvline(best, color="red", ls="--", label=f"best epoch = {best + 1}")
    a.set_title(f"Baseline CNN: {title}"); a.set_xlabel("epoch"); a.legend()
fig.tight_layout(); fig.savefig("reports/figures/fig7_baseline_curves.png", dpi=150)