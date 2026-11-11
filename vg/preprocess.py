"""The decode+resize used at serving time. It matches src/data.py training loads.

Output is float32 in [0, 255]. Scaling lives inside the saved model
(Rescaling for the baseline, MobileNetV2 preprocess for the transfer model).
"""
import tensorflow as tf


def decode_resize(image_bytes, size: int):
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, (size, size), method="bilinear", antialias=True)
    img = tf.cast(img, tf.float32)
    img.set_shape((size, size, 3))
    return img
