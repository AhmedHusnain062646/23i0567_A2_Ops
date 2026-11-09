# labs/toy_mlflow.py   usage: python labs\toy_mlflow.py <lr> <local|dagshub>
import os, sys
import matplotlib.pyplot as plt
import mlflow
import tensorflow as tf
from dotenv import load_dotenv
load_dotenv()

lr, target = float(sys.argv[1]), sys.argv[2]

# if target == "dagshub":
#     from dotenv import load_dotenv
#     load_dotenv()                                   # reads .env, never hard-code secrets
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
# else:
#     mlflow.set_tracking_uri("sqlite:///mlflow_toy.db")
mlflow.set_experiment("toy-fashion-mnist")

tf.keras.utils.set_random_seed(42)
(x, y), (xv, yv) = tf.keras.datasets.fashion_mnist.load_data()
x, xv = x[:10000] / 255.0, xv[:2000] / 255.0
y, yv = y[:10000], yv[:2000]

model = tf.keras.Sequential([
    tf.keras.layers.Input((28, 28)), tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")])
# model.compile(tf.keras.optimizers.Adam(lr), "sparse_categorical_crossentropy", ["accuracy"])
model.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

with mlflow.start_run(run_name=f"toy-lr-{lr}"):
    mlflow.log_params({"learning_rate": lr, "units": 64, "epochs": 5, "batch_size": 64})
    mlflow.set_tags({"lab": "3.1/3.2", "dataset": "fashion-mnist-subset"})
    log_epoch = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda e, logs: mlflow.log_metrics(logs, step=e))
    h = model.fit(x, y, validation_data=(xv, yv), epochs=5, batch_size=64,
                  callbacks=[log_epoch], verbose=2)

    fig, ax = plt.subplots()
    ax.plot(h.history["loss"], label="train"); ax.plot(h.history["val_loss"], label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()
    mlflow.log_figure(fig, "loss_curve.png")        # one plot as an artifact

    try:
        mlflow.keras.log_model(model, name="model")             # MLflow 3.x
    except TypeError:
        mlflow.keras.log_model(model, artifact_path="model")    # MLflow 2.x