# src/train.py
import argparse, json, os, subprocess, sys
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import tensorflow as tf
import yaml
from dotenv import load_dotenv
import data
from data import make_ds
from models import build_baseline, build_mobilenet, unfreeze_last_block
from evaluate import evaluate

load_dotenv()
ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["baseline", "mobilenet"], default="mobilenet")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--ft-epochs", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--ft-lr", type=float, default=1e-5)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--track", choices=["off", "manual", "auto"], default="off")
ap.add_argument("--run-name", default=None, help="defaults to the model name")
ap.add_argument("--dropout", type=float, default=None, help="override dropout (baseline=0.3, mobilenet=0.2)")
ap.add_argument("--img-size", type=int, default=None, help="override image size (default: 128)")
args = ap.parse_args()

SEED = args.seed
tf.keras.utils.set_random_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
if args.img_size is not None:
    data.IMG_SIZE = args.img_size
Path("models").mkdir(exist_ok=True)
Path("reports/figures").mkdir(parents=True, exist_ok=True)

name = args.model
run = args.run_name or name

def sh(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()

# ---------------- MLflow setup ----------------
stack = ExitStack()
if args.track != "off":
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("VisionGuard")
    if args.track == "auto":
        mlflow.tensorflow.autolog(log_models=False, log_datasets=False)
    stack.enter_context(mlflow.start_run(run_name=f"{run}-{args.track}"))
    if args.track == "manual":
        mlflow.log_params({"model": name, "img_size": data.IMG_SIZE, "batch_size": data.BATCH,
                           "epochs_stage1": args.epochs, "epochs_stage2": args.ft_epochs,
                           "lr_stage1": args.lr, "lr_stage2": args.ft_lr, "seed": SEED,
                           "loss": "binary_crossentropy", "optimizer": "adam",
                           "augmentation": "flip,rot0.05,zoom0.1,contrast0.1",
                           "dropout": args.dropout if args.dropout is not None else (
                               0.3 if name == "baseline" else 0.2)})

class EpochLogger(tf.keras.callbacks.Callback):
    """Manual logging: one MLflow metric point per epoch, step continues across stages."""
    def __init__(self, stage, offset=0):
        super().__init__(); self.stage, self.offset = stage, offset
    def on_epoch_end(self, epoch, logs=None):
        m = {k: float(v) for k, v in (logs or {}).items()}
        m["lr"] = float(self.model.optimizer.learning_rate)
        m["stage"] = self.stage
        mlflow.log_metrics(m, step=self.offset + epoch)

train_ds = make_ds("train", training=True, seed=SEED)
val_ds = make_ds("val", training=False)

def compile_model(m, lr):
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                       tf.keras.metrics.Precision(name="precision"),
                       tf.keras.metrics.Recall(name="recall")])

def callbacks(path, stage, offset=0):
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(path, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-7),
    ]
    if args.track == "manual":
        cbs.append(EpochLogger(stage, offset))
    return cbs

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
    return best

_dropout_kw = {} if args.dropout is None else {"dropout": args.dropout}
if name == "baseline":
    model = build_baseline(data.IMG_SIZE, **_dropout_kw)
else:
    model = build_mobilenet(data.IMG_SIZE, **_dropout_kw)
compile_model(model, args.lr)
model.summary()

# ---- stage 1 ----
p1 = f"models/{run}.keras" if name == "baseline" else f"models/{run}_stage1.keras"
h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks(p1, 1))
best1 = plot(h1.history, f"{run} stage 1", f"reports/figures/{run}_stage1_curves.png")
history = {"stage1": h1.history}
best = {"stage1": best1 + 1}
trainable_file = None

# ---- stage 2 (mobilenet only) ----
if name == "mobilenet":
    trainable = unfreeze_last_block(model)
    trainable_file = f"reports/{run}_trainable_layers.txt"
    Path(trainable_file).write_text("\n".join(trainable))
    compile_model(model, args.ft_lr)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.ft_epochs,
                   callbacks=callbacks(f"models/{run}.keras", 2, offset=len(h1.epoch)))
    best2 = plot(h2.history, f"{run} stage 2 (fine-tune)", f"reports/figures/{run}_stage2_curves.png")
    history["stage2"] = h2.history
    best["stage2"] = best2 + 1

hist_path = f"reports/{run}_history.json"
json.dump({s: {k: [float(v) for v in vals] for k, vals in h.items()}
           for s, h in history.items()}, open(hist_path, "w"))

# ---------------- test evaluation + tags + artifacts (both modes) ----------------
if args.track != "off":
    metrics, figs = evaluate(run, model)
    mlflow.log_metrics({f"test_{k}": float(v) for k, v in metrics.items() if k != "model"})
    for s, b in best.items():
        mlflow.log_metric(f"{s}_best_epoch", b)
    last = history["stage2"] if "stage2" in history else history["stage1"]
    mlflow.log_metric("final_val_loss_best", float(min(last["val_loss"])))

    dvc_ver = {}
    for f in ("data/raw.dvc", "data/splits.dvc"):
        dvc_ver[f] = yaml.safe_load(open(f))["outs"][0]["md5"]
    mlflow.set_tags({
        "git_commit": sh("git", "rev-parse", "HEAD"),
        "git_dirty": str(bool(sh("git", "status", "--porcelain", "--untracked-files=no"))),
        "dvc_data_md5": dvc_ver["data/raw.dvc"],
        "dvc_splits_md5": dvc_ver["data/splits.dvc"],
        "seed": str(SEED), "logging_mode": args.track, "model_family": name,
    })

    # 3.4 artifacts
    Path("reports/requirements_frozen.txt").write_text(sh(sys.executable, "-m", "pip", "freeze"))
    for f in figs[:2]:
        mlflow.log_artifact(f, "figures")
    mlflow.log_artifact(figs[2], "error_analysis")
    mlflow.log_artifacts("data/splits", "splits")
    mlflow.log_artifact("reports/requirements_frozen.txt", "environment")
    mlflow.log_artifact(hist_path, "history")
    if trainable_file:
        mlflow.log_artifact(trainable_file, "history")
    try:
        mlflow.keras.log_model(model, name="model")
    except Exception as e:
        print("log_model failed, uploading the .keras file instead:", e)
        mlflow.log_artifact(f"models/{run}.keras", "model_file")
    stack.close()
    print("Tracked run:", run, args.track)