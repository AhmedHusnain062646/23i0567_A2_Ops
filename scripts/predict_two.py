"""Task 4.1 console evidence: classify two known images with the serving module.
Run: MODEL_PATH=models/champion.keras python scripts/predict_two.py real.jpg fake.png"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.inference import LocalPredictor  # noqa: E402

thr = os.getenv("THRESHOLD")
pred = LocalPredictor(os.getenv("MODEL_PATH", "models/champion.keras"), float(thr) if thr else None)
for p in sys.argv[1:]:
    print(p, "->", json.dumps(pred.predict(Path(p).read_bytes())))
