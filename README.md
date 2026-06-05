# Forest Fire Prediction

End-to-end ML project predicting Algerian forest fires using weather & FWI indices. Applying a meticulous, rigorous approach to data preparation and experimentation, this project scales from raw data to a deployed prediction system.

## Overview
This repository houses the complete lifecycle of a production-level machine learning system. It is built with modular Python, DVC for data versioning, MLflow for remote experiment tracking via DagsHub, and sets the foundation for a robust API backend.

## Model Experimentation & Training
The model predicts the Fire Weather Index (FWI) target variable based on critical environmental features. The dataset was split 80/20 for training and testing, and scaled using a robust preprocessing pipeline. 

To ensure the best possible fit without assumptions, 6 distinct algorithms were evaluated as baseline models:
* Linear Regression
* Ridge Regression
* Lasso Regression
* Support Vector Regressor (SVR)
* Random Forest Regressor
* XGBoost Regressor

### Final Tuned Model
Following a rigorous 5-fold `GridSearchCV` testing 35 parameter candidates (totalling 175 fits), **Ridge Regression** emerged as the winning model.
* **Optimal Hyperparameters**: `alpha`: 1.0, `solver`: 'sparse_cg'
* **Test Performance**: Achieved a highly accurate R² Score of 0.9746.

## MLOps: Remote Tracking & Model Registry
This project utilizes **DagsHub** as the remote backend for production MLOps:
* **DVC (Data Version Control)** tracks large datasets (`train.csv`, `test.csv`) and binary artifacts securely in cloud storage.
* **MLflow** tracks all experiment parameters, metrics, and models in a remote tracking server.
* The final optimized model is officially registered in the MLflow Model Registry as **`Forest-Fire-Ridge-Predictor`** (Version 1).

## Pipeline & Deployment
With the experimentation phase successfully locked in, the final optimized model and standard preprocessor are automatically exported to `artifacts/tuned_model.pkl` and `artifacts/preprocessor.pkl`. The project architecture now moves to the Inference phase (`src/Pipeline/Predict_Pipeline.py`) to load these artifacts and generate predictions on new data.

## Quick Start
1. Clone the repo: `git clone https://github.com/batman230-sai/forest-fire-prediction.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Execute the full end-to-end training and tuning pipeline: `dvc repro`
4. Access the MLflow Tracking Server (Local Fallback): `mlflow ui --backend-store-uri sqlite:///mlflow.db`