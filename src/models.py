# # src/models.py
# from tensorflow.keras import layers, models

# def build_baseline(img_size=224, dropout=0.3):
#     inp = layers.Input((img_size, img_size, 3), name="image")
#     x = layers.Rescaling(1 / 255.0, name="rescale")(inp)
#     for i, f in enumerate((32, 64, 128, 128), start=1):
#         x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
#         x = layers.BatchNormalization(name=f"bn{i}")(x)
#         x = layers.ReLU(name=f"relu{i}")(x)
#         x = layers.MaxPooling2D(name=f"pool{i}")(x)
#     x = layers.GlobalAveragePooling2D(name="gap")(x)
#     x = layers.Dropout(dropout, name="dropout")(x)
#     out = layers.Dense(1, activation="sigmoid", name="prob_ai")(x)
#     return models.Model(inp, out, name="baseline_cnn")


# src/models.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_baseline(img_size=128, dropout=0.3):
    inp = layers.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1 / 255.0, name="rescale")(inp)
    for i, f in enumerate((32, 64, 128, 128), start=1):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.ReLU(name=f"relu{i}")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="prob_ai")(x)
    return models.Model(inp, out, name="baseline_cnn")

def build_mobilenet(img_size=128, dropout=0.2):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet")
    base.trainable = False                                  # stage 1: frozen
    inp = layers.Input((img_size, img_size, 3), name="image")
    # MobileNetV2 expects pixels in [-1, 1]; this equals mobilenet_v2.preprocess_input
    x = layers.Rescaling(1 / 127.5, offset=-1, name="mnv2_preprocess")(inp)
    x = base(x, training=False)                             # keep BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="prob_ai")(x)
    return models.Model(inp, out, name="mobilenetv2_transfer")

def unfreeze_last_block(model, start="block_16_expand"):
    """Stage 2: unfreeze only MobileNetV2's last block (BatchNorm layers stay frozen)."""
    base = next(l for l in model.layers if isinstance(l, tf.keras.Model))
    base.trainable = True
    names = [l.name for l in base.layers]
    cut = names.index(start)
    for l in base.layers[:cut]:
        l.trainable = False
    for l in base.layers[cut:]:
        l.trainable = not isinstance(l, layers.BatchNormalization)
    trainable = [l.name for l in base.layers[cut:] if l.trainable and l.weights]
    return trainable + ["gap", "dropout", "prob_ai"]