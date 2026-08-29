# MLOps Assignment 2 — Cats vs Dogs End-to-End MLOps Pipeline

**Course:** MLOps (S1-25_AIMLCZG523)
**Author:** Aswin G (BITS ID: 2024AC05661)

An end-to-end MLOps pipeline for a binary image classifier (Cats vs Dogs)
covering data/model versioning, experiment tracking, packaging,
containerization, CI, CD, and monitoring — built entirely with open-source
tools.

## Architecture at a glance

```
Git + DVC  ──▶  MLflow tracked training  ──▶  model.pt
                                                  │
                                                  ▼
                                     FastAPI inference service
                                                  │
                                                  ▼
                                        Docker image (Dockerfile)
                                                  │
                        GitHub Actions CI: test ─▶ build ─▶ push (GHCR)
                                                  │
                        GitHub Actions CD: pull ─▶ docker compose up ─▶ smoke test
                                                  │
                                    Logs + Prometheus metrics + post-deploy
                                    performance tracking (monitoring/)
```

## Repository layout

```
src/data/          preprocessing + Kaggle download
src/models/        CNN model, training (+MLflow), inference utilities
src/api/           FastAPI service (health, predict, metrics)
tests/             pytest unit tests (preprocessing, inference, API)
deployment/        deployment-target docker-compose, prometheus config, smoke test
monitoring/        post-deployment performance-tracking script
.github/workflows/ ci.yml (M3) and cd.yml (M4)
dvc.yaml           DVC pipeline: preprocess -> train
params.yaml        all pipeline/training hyperparameters
Dockerfile         inference-service image
docker-compose.yml local dev compose (builds from source)
```

---

## M1 — Model Development & Experiment Tracking

### 1. Data & code versioning

* Source code is versioned with **Git** (this repository).
* The dataset is versioned with **DVC**, backed by a local remote
  (`/d/dvc-storage` — swap for S3/GCS/Azure in production by changing
  `dvc remote add`).

Dataset: **Kaggle "dogsVScats" (salader/dogsvscats)**, downloaded with
the official Kaggle API.

```bash
# one-time: put your kaggle.json at ~/.kaggle/kaggle.json
python -m src.data.download --dest data/raw
dvc add data/raw
git add data/raw.dvc .gitignore
git commit -m "Track raw Kaggle dataset with DVC"
dvc push       # uploads data to the DVC remote
```

Pre-processing (resize to 224x224 RGB, 80/10/10 train/val/test split) is
also DVC-tracked as a pipeline stage, so it's fully reproducible:

```bash
dvc repro preprocess
git add data/processed.dvc dvc.lock
git commit -m "Track pre-processed data"
```

> `params.yaml -> data.max_per_class` caps images per class (default 1500)
> so training stays fast on a laptop CPU. Raise it (or remove the cap) for
> a fuller run — DVC will detect the param change and know to re-run.

### 2. Model building

A baseline CNN (`src/models/model.py: SimpleCNN`) — 4 conv blocks +
2 FC layers, trained with `BCEWithLogitsLoss`. Saved as a standard
PyTorch state-dict at `models/model.pt`.

### 3. Experiment tracking (MLflow)

`src/models/train.py` logs every run to MLflow: all hyperparameters,
per-epoch train/val loss and val accuracy, final test accuracy/precision/
recall/F1, the loss-curve plot, the confusion-matrix plot, and the model
artifact itself.

```bash
dvc repro train
# or directly:
python -m src.models.train --params params.yaml

# inspect runs:
mlflow ui --backend-store-uri mlruns   # http://localhost:5000
```

---

## M2 — Model Packaging & Containerization

### 1. Inference service

`src/api/main.py` (FastAPI) exposes:

* `GET /health` — liveness/readiness probe (`{"status": "ok", "model_loaded": true}`)
* `POST /predict` — multipart image upload → `{"label": "cat"|"dog", "probabilities": {...}, "latency_ms": ...}`
* `GET /metrics` — Prometheus exposition (see M5)

Run it locally without Docker:

```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

### 2. Environment specification

`requirements.txt` pins every runtime dependency (incl. CPU-only PyTorch
wheels via `--extra-index-url`) for reproducible builds.
`requirements-dev.txt` adds training/tooling-only deps (MLflow, DVC,
pytest, matplotlib, kaggle) that are **not** shipped in the inference image.

### 3. Containerization

```bash
docker compose build
docker compose up -d
curl -f http://localhost:8080/health
curl -F "file=@some_cat_photo.jpg" http://localhost:8080/predict
```

(Or plain Docker: `docker build -t catsdogs-api:local . && docker run -p 8080:8000 catsdogs-api:local`.)

---

## M3 — CI Pipeline (`.github/workflows/ci.yml`)

On every push and pull request:

1. **test** job — checkout, set up Python 3.12, install `requirements-dev.txt`, run `pytest` (`tests/test_preprocess.py` covers the pre-processing function, `tests/test_predict.py`/`tests/test_api.py` cover the inference path).
2. **build-and-push** job (needs `test`) — builds the Docker image, runs the same `deployment/smoke_test.py` against a throwaway container as an extra CI gate, then — **only on a push to `main`** — pushes `ghcr.io/<owner>/<repo>:latest` and `:<sha>` to **GitHub Container Registry**.

---

## M4 — CD Pipeline & Deployment (`.github/workflows/cd.yml`)

Deployment target: **Docker Compose** (`deployment/docker-compose.yml`),
standing in for "a simple VM server".

The CD workflow triggers via `workflow_run` once CI finishes successfully
on `main`, then:

1. Logs in to GHCR and pulls the freshly published image.
2. `docker compose up -d` (deploy/update the running service).
3. Runs `deployment/smoke_test.py` (health check + one real prediction call) — **the job fails if the smoke test fails**, blocking a bad deploy.
4. Tears the container down again (GitHub-hosted runners are ephemeral — see below for a persistent local run).

### Running the "deployment target" persistently yourself

For the screen recording, run the same compose file on your own machine
(the assignment's "simple VM server") so the service stays up between the
CI run and your `curl` demo:

```bash
cd deployment
echo "DEPLOY_IMAGE=ghcr.io/<owner>/<repo>:latest" > .env
docker pull ghcr.io/<owner>/<repo>:latest
docker compose up -d
python smoke_test.py --base-url http://localhost:8080
```

This also starts a **Prometheus** container scraping the API's `/metrics`
endpoint (http://localhost:9091).

---

## M5 — Monitoring, Logs & Final Submission

### 1. Basic monitoring & logging

* Every request is logged as a structured JSON line (`src/api/main.py`,
  `log_requests` middleware) — method, path, status, latency; `/predict`
  additionally logs filename, content-type, predicted label and
  probabilities. **Raw image bytes are never logged.**
* `GET /metrics` exposes Prometheus counters/histograms:
  `http_requests_total`, `http_request_latency_seconds`,
  `predictions_total{label=...}`.

### 2. Model performance tracking (post-deployment)

`monitoring/simulate_requests.py` samples a batch of labeled test images,
sends each to the live `/predict` endpoint, and compares predictions
against ground truth:

```bash
python monitoring/simulate_requests.py \
  --base-url http://localhost:8080 \
  --test-dir data/processed/test \
  --n-per-class 25
```

Outputs: `reports/post_deployment_requests.csv` (per-request log) and
`reports/post_deployment_summary.json` (live accuracy + latency stats) —
so model performance can be tracked over time against training-time
metrics from MLflow.

---

## Running everything end-to-end (for the screen recording)

```bash
# M1
python -m src.data.download --dest data/raw
dvc repro
mlflow ui --backend-store-uri mlruns &

# M2
docker compose build && docker compose up -d
curl -f http://localhost:8080/health
curl -F "file=@sample.jpg" http://localhost:8080/predict

# M3/M4 — push to GitHub, watch Actions tab: CI -> CD
git push origin main

# M5
python monitoring/simulate_requests.py --n-per-class 20
curl http://localhost:8080/metrics
```

## Deliverables checklist

- [x] Git + DVC versioned source, config, data pointers
- [x] Baseline CNN, serialized as `.pt`, MLflow-tracked
- [x] FastAPI service with health + predict endpoints
- [x] `requirements.txt` (pinned) + `Dockerfile`
- [x] Unit tests (`pytest`) for pre-processing + inference
- [x] CI: test -> build -> push to GHCR (GitHub Actions)
- [x] CD: pull -> deploy (Docker Compose) -> smoke test (GitHub Actions)
- [x] Logging + Prometheus metrics + post-deployment performance tracking
- [ ] Zip the repo (excluding `.venv/`, `mlruns/` optional) for submission
- [ ] Record < 5 min screen capture: code change -> CI -> CD -> live prediction
