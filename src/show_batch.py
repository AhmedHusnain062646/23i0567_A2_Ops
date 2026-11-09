# src/show_batch.py
import matplotlib.pyplot as plt
from data import make_ds

ds = make_ds("train", training=True, batch=16)
imgs, labels = next(iter(ds))
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for ax, im, lb in zip(axes.ravel(), imgs, labels):
    ax.imshow(im.numpy().astype("uint8")); ax.axis("off")
    ax.set_title("AI" if lb.numpy() == 1 else "Real", fontsize=9)
fig.suptitle("One augmented training batch (224x224)")
fig.tight_layout()
fig.savefig("reports/figures/fig6_augmented_batch.png", dpi=150)