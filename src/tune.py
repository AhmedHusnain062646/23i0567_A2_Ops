# src/tune.py  -- Keras Tuner random search on the MobileNetV2 model (task 2.7, apply part)
from pathlib import Path
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
import data
from data import make_ds
from models import build_mobilenet

SEED = 42
tf.keras.utils.set_random_seed(SEED)
Path("reports").mkdir(exist_ok=True)

train_ds = make_ds("train", training=True, seed=SEED)
val_ds = make_ds("val", training=False)

def build(hp):
    dropout = hp.Float("dropout", 0.1, 0.6, step=0.1)
    lr = hp.Float("lr", 1e-5, 1e-2, sampling="log")
    n_unfrozen = hp.Choice("unfrozen_layers", [0, 10, 20, 30])   # last N backbone layers

    model = build_mobilenet(data.IMG_SIZE, dropout)
    base = next(l for l in model.layers if isinstance(l, tf.keras.Model))
    if n_unfrozen > 0:
        base.trainable = True
        for l in base.layers[:-n_unfrozen]:
            l.trainable = False
        for l in base.layers[-n_unfrozen:]:
            l.trainable = not isinstance(l, tf.keras.layers.BatchNormalization)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model

tuner = kt.RandomSearch(build, objective="val_loss", max_trials=8, seed=SEED,
                        directory="tuner_vg", project_name="mobilenet", overwrite=True)
tuner.search(train_ds, validation_data=val_ds, epochs=2, verbose=2)

tuner.results_summary()
rows = []
for t in tuner.oracle.trials.values():
    if t.score is None:
        continue
    rows.append({**t.hyperparameters.values, "val_loss": t.score,
                 "val_accuracy": t.metrics.get_best_value("val_accuracy")})
df = pd.DataFrame(rows).sort_values("val_loss").reset_index(drop=True)
df.to_csv("reports/tuner_results.csv", index=False)
print("\n", df.round(4).to_string())

hp_cols = ["dropout", "lr", "unfrozen_layers"]
print("\nSpearman correlation of each setting with val_loss (closer to +-1 = mattered more):")
print(df[hp_cols + ["val_loss"]].corr(method="spearman")["val_loss"][hp_cols].round(2))
print("\nBest:", tuner.get_best_hyperparameters(1)[0].values)