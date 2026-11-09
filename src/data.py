# src/data.py
import pandas as pd
import tensorflow as tf

IMG_SIZE = 224
BATCH = 32
AUTOTUNE = tf.data.AUTOTUNE

def _load(path, label):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)  # forces RGB, handles L/P/RGBA
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear",
                          antialias=True)   # stretches to square, see note below
    img = tf.cast(img, tf.float32)          # values 0-255, preprocessing happens in the model
    return img, tf.cast(label, tf.float32)

augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
], name="augmentation")

def make_ds(split, training, batch=BATCH, seed=42):
    df = pd.read_csv(f"data/splits/{split}.csv")
    ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["label"].values))
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch).prefetch(AUTOTUNE)