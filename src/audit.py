# src/audit.py
import hashlib, sys
from pathlib import Path
import pandas as pd
from PIL import Image

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()

rows = []
for p in sorted(ROOT.rglob("*")):
    if not p.is_file() or p.suffix.lower() not in EXTS:
        continue
    row = {"path": str(p), "class": p.parent.name,
           "file_size": p.stat().st_size, "sha256": sha256(p),
           "width": None, "height": None, "mode": None, "format": None,
           "corrupt": False}
    try:
        with Image.open(p) as im:
            im.verify()                      # cheap integrity check
        with Image.open(p) as im:            # reopen: verify() invalidates the handle
            im.load()                        # full decode catches truncated files
            row.update(width=im.width, height=im.height,
                       mode=im.mode, format=im.format)
    except Exception as e:
        row["corrupt"] = True
        row["error"] = str(e)
    rows.append(row)

df = pd.DataFrame(rows)
df["is_duplicate"] = df.duplicated("sha256", keep="first")
df.to_csv("reports/audit.csv", index=False)

ok = df[~df.corrupt]
print("Counts per class:\n", df["class"].value_counts(), "\n")
print("Corrupt files:", int(df.corrupt.sum()))
print("Exact duplicates (extra copies):", int(df.is_duplicate.sum()))
print(ok.groupby("class")[["width", "height", "file_size"]]
        .agg(["min", "median", "max"]).T)
print("\nColour modes:\n", ok.groupby(["class", "mode"]).size())
print("\nFormats:\n", ok.groupby(["class", "format"]).size())

# duplicates that cross classes are a label-noise red flag
dups = df[df.duplicated("sha256", keep=False)]
cross = dups.groupby("sha256")["class"].nunique()
print("Duplicate groups spanning both classes:", int((cross > 1).sum()))