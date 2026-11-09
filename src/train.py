# src/train.py
import argparse, json, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import data
from data import make_ds
from models import build_baseline, build_mobilenet, unfreeze_last_block

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["baseline", "mobilenet"], default="mobilenet")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--ft-epochs", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--ft-lr", type=float, default=1e-5)
args = ap.parse_args()

SEED = 42
tf.keras.utils.set_random_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
Path("models").mkdir(exist_ok=True)
Path("reports/figures").mkdir(parents=True, exist_ok=True)

train_ds = make_ds("train", training=True, seed=SEED)
val_ds = make_ds("val", training=False)

def compile_model(m, lr):
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                       tf.keras.metrics.Precision(name="precision"),
                       tf.keras.metrics.Recall(name="recall")])

def callbacks(path):
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2,
                                         restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(path, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=1, min_lr=1e-7),
    ]

def plot(h, title, fname):
    best = int(np.argmin(h["val_loss"]))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, m in zip(ax, ["loss", "accuracy"]):
        a.plot(h[m], label="train"); a.plot(h[f"val_{m}"], label="validation")
        a.axvline(best, color="red", ls="--", label=f"best epoch = {best + 1}")
        a.set_title(f"{title}: {m}"); a.set_xlabel("epoch"); a.legend()
    fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig)
    print(f"{title}: best epoch {best + 1}, val_loss {h['val_loss'][best]:.4f}, "
          f"val_acc {h['val_accuracy'][best]:.4f}")

name = args.model
model = build_baseline(data.IMG_SIZE) if name == "baseline" else build_mobilenet(data.IMG_SIZE)
compile_model(model, args.lr)
model.summary()

# ---- stage 1 (baseline: the only stage; mobilenet: frozen backbone, new head) ----
p1 = "models/baseline.keras" if name == "baseline" else "models/mobilenet_stage1.keras"
h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks(p1))
plot(h1.history, f"{name} stage 1", f"reports/figures/{name}_stage1_curves.png")
history = {"stage1": h1.history}

# ---- stage 2 (mobilenet only): unfreeze last block, much smaller learning rate ----
if name == "mobilenet":
    trainable = unfreeze_last_block(model)
    Path("reports/mobilenet_trainable_layers.txt").write_text("\n".join(trainable))
    print("Trainable layers in stage 2:", trainable)
    compile_model(model, args.ft_lr)                      # recompile after changing trainable flags
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.ft_epochs,
                   callbacks=callbacks("models/mobilenet.keras"))
    plot(h2.history, "mobilenet stage 2 (fine-tune)", "reports/figures/mobilenet_stage2_curves.png")
    history["stage2"] = h2.history

json.dump({s: {k: [float(v) for v in vals] for k, vals in h.items()}
           for s, h in history.items()}, open(f"reports/{name}_history.json", "w"))