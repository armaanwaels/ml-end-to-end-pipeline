
# ML End-to-End Pipeline: Credit Card Fraud Detection

This repository contains a complete machine learning workflow for credit card fraud detection, including data preprocessing, model training, evaluation with tracked metrics, and a production-ready FastAPI microservice for real-time prediction. The pipeline is fully versioned with **DVC** to handle data, artifacts, and reproducibility.

---

## Project Overview

* **End-to-end ML pipeline:** Preprocessing, training, evaluation, and deployment.
* **Experiment tracking with DVC:** Data, artifacts, metrics, and params are versioned.
* **Model serving via FastAPI** for real-time inference.
* **Dockerized** for portability and reproducibility.
* **Cloud deployment** using Render.com.
* **Clean, modular, and easy to maintain codebase.**

---

## Dataset

* **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
* The dataset is not included in this repository due to size constraints.
* Use **DVC** to pull datasets and artifacts if you have access to the remote storage.

---

## System Architecture

1. **Pipeline stages managed by DVC** (`dvc.yaml`):

   * `preprocess` → raw → processed dataset
   * `train` → model + scaler artifacts
   * `evaluate` → evaluation metrics JSON
2. **Model serialization** using joblib (`artifacts/model.joblib`, `artifacts/scaler.joblib`)
3. **FastAPI microservice** for inference (`app/main.py`)
4. **Docker** for easy deployment
5. **Render.com** for live hosting

---

## Getting Started

### Local Development

```bash
git clone https://github.com/armaanwaels/ml-end-to-end-pipeline.git
cd ml-end-to-end-pipeline

# create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# reproduce the pipeline
dvc repro

# show metrics
dvc metrics show
```

Start the FastAPI app:

```bash
cd app
uvicorn main:app --reload
```

The API documentation will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### Docker

```bash
cd app
docker build -t credit-fraud-api .
docker run -p 8000:8000 credit-fraud-api
```

The API documentation will be available at [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Live Demo

A live version of the API is available here:
[https://ml-end-to-end-pipeline.onrender.com/docs](https://ml-end-to-end-pipeline.onrender.com/docs)

---

## API Usage

### `POST /predict`

**Request body (JSON):**

```json
{
  "features": [value1, value2, ..., valueN]
}
```

**Response:**

```json
{
  "prediction": 0,
  "fraud_probability": 0.0153
}
```

You can test the API interactively using the `/docs` Swagger UI.

---

## Repository Structure

```
notebooks/                # Exploratory analysis and experiments
src/                      # Pipeline scripts (preprocess, train, evaluate)
artifacts/                # Trained model + scaler (tracked by DVC)
data/
  ├─ raw/                 # Raw dataset (DVC-tracked)
  └─ processed/           # Processed dataset (DVC-tracked)
metrics/                  # JSON metrics tracked by DVC
app/
  ├─ main.py              # FastAPI application
  ├─ requirements.txt     # API requirements
  └─ Dockerfile
dvc.yaml                  # Pipeline definition
dvc.lock                  # Pipeline state
params.yaml               # Parameters (train/eval configs)
requirements.txt          # Project-level requirements
.gitignore
README.md
```

---

## Deployment Notes

* Fully containerized and cloud-deployable.
* Data and artifacts tracked with **DVC** (not in Git).
* Metrics (`metrics/*.json`) and parameters (`params.yaml`) enable reproducible experiments.
* API follows the same pipeline as in development for consistency.

---

## Author

**Armaan Waels**
[GitHub](https://github.com/armaanwaels)

---
