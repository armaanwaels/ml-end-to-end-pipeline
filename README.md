
# ML End-to-End Pipeline: Credit Card Fraud Detection

This repository contains a complete machine learning workflow for credit card fraud detection, including data analysis, model training, evaluation, and a production-ready FastAPI microservice for real-time prediction.

---

## Project Overview

- **End-to-end ML pipeline:** EDA, feature engineering, model training, evaluation, and deployment.
- **Model serving via FastAPI**
- **Dockerized** for portability and reproducibility.
- **Cloud deployment** using Render.com.
- **Clean, modular, and easy to maintain codebase.**

---

## Dataset

- **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- The dataset is not included due to size constraints.

---

## System Architecture

1. **Jupyter notebooks** for exploratory analysis and model development (`notebooks/`)
2. **Model serialization** using joblib (`app/model.joblib`, `app/scaler.joblib`)
3. **FastAPI microservice** for inference (`app/main.py`)
4. **Docker** for easy deployment
5. **Render.com** for live hosting

---

## Getting Started

### Local Development

```bash
git clone https://github.com/armaanwaels/ml-end-to-end-pipeline.git
cd ml-end-to-end-pipeline/app

# create a virtual environment
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# Start FastAPI
uvicorn main:app --reload
````

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
notebooks/             # Data analysis and model training
app/
  ├─ main.py           # FastAPI application
  ├─ model.joblib      # Trained model
  ├─ scaler.joblib     # Data scaler
  ├─ requirements.txt
  └─ Dockerfile
requirements.txt       # Project-level requirements
.gitignore
README.md
```

---

## Deployment Notes

* Fully containerized and cloud-deployable.
* All inference artifacts are included in `app/`.
* API follows the same pipeline as in development for consistency.

---

## Author

**Armaan Waels**
[GitHub](https://github.com/armaanwaels)

```

---

**This will render perfectly on GitHub. Paste the whole thing, and you’ll have bolded sections, proper headings, and correct formatting for code and command line blocks.**
```
