# labs/toy_tuner.py  -- random search on Fashion-MNIST: learning rate and units
import keras_tuner as kt
import tensorflow as tf

(x, y), (xv, yv) = tf.keras.datasets.fashion_mnist.load_data()
x = (x[:10000] / 255.0).astype("float32"); y = y[:10000].astype("int32")
xv = (xv[:2000] / 255.0).astype("float32"); yv = yv[:2000].astype("int32")

def build(hp):
    m = tf.keras.Sequential([
        tf.keras.layers.Input((28, 28)), tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hp.Int("units", 32, 256, step=32), activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")])
    # m.compile(tf.keras.optimizers.Adam(hp.Float("lr", 1e-4, 1e-2, sampling="log")),
    #           "sparse_categorical_crossentropy", ["accuracy"])
    m.compile(optimizer=tf.keras.optimizers.Adam(hp.Float("lr", 1e-4, 1e-2, sampling="log")),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

tuner = kt.RandomSearch(build, objective="val_accuracy", max_trials=6,
                        directory="tuner_toy", project_name="fmnist", seed=42, overwrite=True)
tuner.search(x, y, validation_data=(xv, yv), epochs=3, verbose=2)
tuner.results_summary()
print("Best:", tuner.get_best_hyperparameters(1)[0].values)

print("x:", x.dtype)
print("y:", y.dtype)
print("xv:", xv.dtype)
print("yv:", yv.dtype)