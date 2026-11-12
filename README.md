# VisionGuard: AI-generated vs real image classification

[![CI](https://github.com/AhmedHusnain062646/23i0567_A2_Ops/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmedHusnain062646/23i0567_A2_Ops/actions/workflows/ci.yml)

VisionGuard is a web service that tells whether an uploaded image is AI-generated or real.
A TensorFlow/Keras model is trained and tracked with MLflow on DagsHub, versioned with DVC,
served through a Flask app, logged to a database, containerised with Docker Compose, and
the model API is deployed to the cloud.

| | |
|---|---|
| GitHub | https://github.com/AhmedHusnain062646/23i0567_A2_Ops |
| DagsHub | https://dagshub.com/ahmed0567/23i0567_A2_Ops |
| Live model API | https://two3i0567-a2-ops.onrender.com |

## Repository layout

| Path | Purpose |
|---|---|
| `src/` | Training (`train.py`), evaluation, MLflow registry scripts (`register_model.py`, `load_champion.py`, `export_champion.py`) |
| `app/` | Flask web app: upload, result, history, health check; inference module; database models |
| `model_api/` | Stand-alone model API that is deployed to the cloud |
| `models/` | Trained Keras models and `champion_meta.json` (threshold, version) |
| `data/` | Dataset and split lists (tracked with DVC, not Git) |
| `reports/` | Audit table, metrics, figures, run comparison |
| `tests/` | Unit tests (run in CI) |
| `.github/workflows/ci.yml` | CI: install, lint, test, build Docker image |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (PowerShell)
pip install -r requirements.txt
cp .env.example .env              # then fill in the values; .env is git-ignored
dvc pull                          # fetch the dataset from the DagsHub DVC remote
```

Secrets (MLflow token, API key, database password) are read from environment
variables / `.env` and are never committed.

## Training and experiment tracking

```bash
python src/train.py --model mobilenet --track manual --run-name mobilenet_manual
python src/train.py --model mobilenet --track auto   --run-name mobilenet_auto
```

Every run logs parameters, per-epoch metrics, test metrics, tags (git commit,
DVC data version, seed) and artifacts (confusion matrix, ROC curve, split lists,
package list, model) to MLflow on DagsHub.

## Model registry

```bash
python src/register_model.py --run-id <run_id> --alias champion
python src/load_champion.py path/to/image.jpg     # loads VisionGuard@champion and classifies one image
python src/export_champion.py                     # writes models/champion.keras + champion_meta.json
```

## Running the Flask app (with database)

```bash
docker compose up -d --build
```

Open http://localhost:5000 (see `docker-compose.yml` for the published port).
The database service is not exposed outside the Compose network.

## Model API

```bash
docker compose -f docker-compose.api.yml up -d --build
```

The API requires the secret key from `REMOTE_API_KEY` / `API_KEY`. The cloud
deployment is described in the project report.

## Tests and CI

```bash
pytest
ruff check .
```

GitHub Actions runs the same checks on every push and builds the Docker image.
