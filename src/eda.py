# src/eda.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

SEED = 42
OUT = Path("reports/figures"); OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("reports/audit.csv")
df = df[~df.corrupt]                         # drop unreadable files
clean = df[~df.is_duplicate].copy()          # drop exact duplicate copies
clean.to_csv("reports/clean.csv", index=False)   # manifest used later for splitting
print("kept", len(clean), "of", len(df), "readable images")

classes = sorted(clean["class"].unique())
colors = ["tab:blue", "tab:orange"]

# ---- Figure 1: sample grid per class (6 x 4 per class) ----
rng = np.random.default_rng(SEED)
for cls in classes:
    paths = clean[clean["class"] == cls]["path"].tolist()
    picks = rng.choice(paths, size=min(24, len(paths)), replace=False)
    fig, axes = plt.subplots(4, 6, figsize=(14, 9.5))
    for ax, p in zip(axes.ravel(), picks):
        with Image.open(p) as im:
            ax.imshow(im.convert("RGB"))
        ax.axis("off")
    fig.suptitle(f"Random samples: {cls}", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / f"fig1_samples_{cls}.png", dpi=150)
    plt.close(fig)

# ---- Figures 2-4: histograms per class ----
def hist(col, title, xlabel, fname, log_x=False):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [clean[clean["class"] == c][col].dropna() for c in classes]
    allv = pd.concat(data)
    bins = (np.logspace(np.log10(max(allv.min(), 1)), np.log10(allv.max()), 40)
            if log_x else 40)
    for d, c, col_ in zip(data, classes, colors):
        ax.hist(d, bins=bins, alpha=0.55, label=f"{c} (n={len(d)})", color=col_)
    if log_x:
        ax.set_xscale("log")
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("Number of images")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150); plt.close(fig)

hist("width",  "Image width per class",  "Width (px)",  "fig2_width.png",  log_x=True)
hist("height", "Image height per class", "Height (px)", "fig3_height.png", log_x=True)
hist("file_size", "File size per class", "File size (bytes)", "fig4_filesize.png", log_x=True)

# ---- Extra: aspect ratio + format, useful for the shortcut-learning discussion ----
clean["aspect"] = clean["width"] / clean["height"]
hist_df = clean.copy()
fig, ax = plt.subplots(figsize=(8, 4.5))
for c, col_ in zip(classes, colors):
    ax.hist(hist_df[hist_df["class"] == c]["aspect"], bins=40, alpha=0.55, label=c, color=col_)
ax.set_title("Aspect ratio (width / height) per class")
ax.set_xlabel("Aspect ratio"); ax.set_ylabel("Number of images"); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "fig5_aspect.png", dpi=150); plt.close(fig)

print("\nWidth/height/file-size summary per class:")
print(clean.groupby("class")[["width", "height", "file_size", "aspect"]]
      .describe().T.to_string())
print("\nFormat share per class:")
print(pd.crosstab(clean["class"], clean["format"], normalize="index").round(3))