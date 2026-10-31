# Sentinel — Fraud Detection

An end-to-end machine-learning project that detects fraudulent mobile-money transactions. It covers data exploration, leakage-safe feature engineering, XGBoost training, Optuna + MLflow tuning, SHAP explainability, and a FastAPI service with a web console for scoring transactions.

---

## Table of contents
1. [Overview](#overview)
2. [Project structure](#project-structure)
3. [Pipeline](#pipeline)
4. [Features](#features)
5. [Model and results](#model-and-results)
6. [Setup](#setup)
7. [Running the app](#running-the-app)
8. [API reference](#api-reference)
9. [Web console](#web-console)
10. [Configuration](#configuration)
11. [Design notes](#design-notes)
12. [Limitations and next steps](#limitations-and-next-steps)

---

## Overview

| | |
|---|---|
| **Problem** | Binary classification: is a transaction fraudulent? |
| **Model** | XGBoost, tuned with Optuna, tracked with MLflow |
| **Serving** | FastAPI + a single-page HTML console |
| **Explainability** | Per-prediction SHAP contributions (top 3 reasons) |

Because fraud is rare, accuracy is not used. The project evaluates with **PR-AUC, ROC-AUC, precision, recall, F1** and the confusion matrix.

---

## Project structure

```
fraud_detection/
├── app/                         # Serving layer
│   ├── main.py                  # FastAPI app (endpoints, model loading, static frontend)
│   ├── fraud_predictor.py       # FraudPredictor: feature building, scoring, SHAP, history
│   ├── schemas.py               # Pydantic request/response models
│   └── static/index.html        # Web console (Sentinel UI)
├── models/                      # Trained artifacts
│   ├── xgb_fraud.json           # Baseline model
│   ├── model_meta.json          # Baseline metadata (features, threshold, metrics)
│   ├── xgb_fraud_tuned.json     # Tuned model (default for serving)
│   ├── model_meta_tuned.json    # Tuned metadata
│   ├── optuna_trials.csv        # All Optuna trials
│   └── validation_results.csv
├── notebooks/
│   ├── data_understanding.ipynb
│   ├── data_transformation.ipynb
│   ├── feature_engineering.ipynb
│   ├── model_training.ipynb
│   ├── tuning_optuna_mlflow.ipynb
│   ├── shap_analysis.ipynb
│   ├── 07_oop_fraud_predictor.ipynb
│   └── fraud_predictor.py       # Notebook copy of the predictor class
├── mlflow.db                    # MLflow tracking store (SQLite)
├── requirements.txt
└── README.md
```

---

## Pipeline

Run the notebooks in order:

| # | Notebook | Purpose |
|---|---|---|
| 1 | `data_understanding.ipynb` | Explore the data and distributions |
| 2 | `data_transformation.ipynb` | Clean the data, fix dtypes, quality checks |
| 3 | `feature_engineering.ipynb` | Build leakage-safe features, define `STRICT` / `STANDARD` sets |
| 4 | `model_training.ipynb` | Time-based split, baseline XGBoost, threshold selection |
| 5 | `tuning_optuna_mlflow.ipynb` | Optuna hyperparameter search logged to MLflow |
| 6 | `shap_analysis.ipynb` | Global and local feature-importance analysis |
| 7 | `07_oop_fraud_predictor.ipynb` | Package the logic into the `FraudPredictor` class |

**Time-based split** (no random shuffling, to avoid look-ahead leakage):
train up to step 323, validation up to step 377, test afterwards.

---

## Features

Only information available **at decision time** is used. Post-transaction balances (`newbalanceOrig`, `newbalanceDest`) are deliberately excluded because they leak the label.

| Feature | Description |
|---|---|
| `log_amount` | `log1p(amount)` |
| `is_transfer` | 1 for `TRANSFER`, 0 for `CASH_OUT` |
| `hour` | `step % 24` |
| `is_night` | 1 if hour is 0–5 |
| `dest_prior_txn_count` | Number of transactions the destination received in strictly **earlier** steps |
| `amount_to_oldbalance_ratio` | `amount / (oldbalanceOrg + 1)` |
| `sender_near_empty` | Amount ≥ 95% of sender's balance (account drained) |
| `sender_zero_balance` | Sender balance is 0 |

Feature sets:
- **STRICT:** the first five features (no sender-balance information).
- **STANDARD:** all eight. **This is the set used in production.**

---

## Model and results

The tuned XGBoost model (`models/xgb_fraud_tuned.json`) is loaded by default. Key hyperparameters: `max_depth=8`, `learning_rate≈0.024`, `scale_pos_weight≈6.28`, `eval_metric=aucpr`, early stopping 50 rounds. The decision threshold is **0.216**, chosen on the validation set.

| Metric | Validation | Test |
|---|---|---|
| PR-AUC | 0.9436 | **0.9845** |
| ROC-AUC | 0.9999 | 0.9983 |
| Precision | 0.940 | **0.988** |
| Recall | 0.839 | **0.858** |
| F1 | 0.887 | 0.918 |
| TP / FP / FN | 470 / 30 / 90 | 3,441 / 42 / 569 |

For comparison, the baseline model reached a test PR-AUC of 0.9871 with precision ≈ 0.987 and recall ≈ 0.850 (see `models/model_meta.json`). Tuning mostly improved validation PR-AUC (0.939 → 0.944) and recall.

**Risk bands** returned by the API:
- `HIGH`: probability ≥ threshold (flagged as fraud)
- `MEDIUM`: probability ≥ 50% of the threshold
- `LOW`: otherwise

---

## Setup

Requirements: Python 3.10+ (the repo contains bytecode from 3.10 and 3.12).

```bash
cd fraud_detection
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` covers serving (`fastapi`, `uvicorn`, `xgboost`, `pandas`, `numpy`). To re-run the notebooks you will also need `jupyter`, `scikit-learn`, `optuna`, `mlflow`, `shap`, `pyarrow` and `matplotlib`.

---

## Running the app

From the project root:

```bash
uvicorn app.main:app --reload
```

- Web console: http://127.0.0.1:8000
- Interactive API docs (Swagger): http://127.0.0.1:8000/docs

To browse the MLflow experiments:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## API reference

| Method | Path | Description |
|---|---|---|
| GET | `/` | Web console |
| GET | `/health` | `{"status": "ok"}` or `"model_not_loaded"` |
| GET | `/model` | Model name, feature set, features, threshold, metrics, destinations tracked |
| POST | `/predict` | Score one transaction |
| POST | `/predict/batch` | Score up to 1,000 transactions |

`POST /predict` accepts the query parameters `explain` (default `true`, returns SHAP reasons) and `record_history` (default `false`, counts the transaction toward the destination's history).

**Request**

```json
{
  "step": 10,
  "type": "TRANSFER",
  "amount": 181.0,
  "oldbalanceOrg": 181.0,
  "nameDest": "C553264065"
}
```

**Response**

```json
{
  "fraud_probability": 0.97,
  "is_fraud": true,
  "risk_band": "HIGH",
  "threshold": 0.216,
  "model_name": "xgboost-tuned",
  "feature_set": "STANDARD",
  "reasons": [
    {"feature": "sender_near_empty", "value": 1.0, "contribution": 2.4}
  ]
}
```

(Values above are illustrative.) `contribution` is a SHAP value in log-odds; positive pushes toward fraud.

**Example**

```bash
curl -X POST "http://127.0.0.1:8000/predict?explain=true" \
  -H "Content-Type: application/json" \
  -d '{"step":10,"type":"TRANSFER","amount":181,"oldbalanceOrg":181,"nameDest":"C553264065"}'
```

**Validation:** `type` must be `TRANSFER` or `CASH_OUT`; `amount` and `oldbalanceOrg` must be ≥ 0; `step` must be ≥ 0. Invalid input returns HTTP 422. If the model failed to load, endpoints return 503.

---

## Web console

A single-page dark "security console" at `/` (`app/static/index.html`):

- Transaction form with a TRANSFER / CASH_OUT toggle and **suspicious** / **normal** sample buttons
- Animated circular risk dial with a threshold marker
- Risk-band tag and a SHAP "why" bar chart
- Live feed of the last 10 checks
- Model status chip showing recall and precision, plus a light/dark theme toggle

---

## Configuration

| Environment variable | Default | Meaning |
|---|---|---|
| `FRAUD_MODEL_DIR` | `./models` | Folder containing `xgb_fraud*.json` and `model_meta*.json` |
| `FRAUD_USE_TUNED` | `1` | `1` = tuned model, `0` = baseline model |

```bash
FRAUD_USE_TUNED=0 uvicorn app.main:app --reload
```

---

## Design notes

- **Leakage prevention:** time-based splits, no post-transaction balances, and `dest_prior_txn_count` counts only strictly earlier steps. Serving applies the same rule via `DestinationHistory`, which stores counts per step.
- **Threshold tuning:** the threshold is selected on the validation set to balance precision and recall, not left at 0.5.
- **Training/serving parity:** `FraudPredictor.build_features` rebuilds the training features exactly from raw input.
- **Thread safety:** a lock guards the shared in-memory destination history.
- **Explainability:** SHAP contributions come from XGBoost's native `pred_contribs`.

---

## Limitations and next steps

- Destination history is **in-memory** and starts empty on each restart, so `dest_prior_txn_count` is 0 for unseen accounts. Production would use Redis or a feature store.
- Real-world performance should be validated on production data before deployment.
- The API has no authentication or rate limiting.
- The frontend loads fonts from Google Fonts, so it needs internet access for the intended typography.
- Possible next steps: Dockerfile, unit tests for `FraudPredictor`, model monitoring and drift detection, a persistent feature store, and a CI pipeline.
