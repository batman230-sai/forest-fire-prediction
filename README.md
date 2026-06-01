# Forest Fire Prediction

End-to-end ML project predicting Algerian forest fires using weather & FWI indices. Applying a meticulous, rigorous approach to data preparation and experimentation, this project scales from raw data to a deployed prediction system in 30 days.

## Overview
This repository houses the complete lifecycle of a production-level machine learning system. It is built with modular Python, a FastAPI backend, a Flutter app, MLflow tracking, Docker & CI/CD.

## Model Experimentation & Training
The model was designed to predict the Fire Weather Index (FWI) target variable based on 10 features, including Temperature, Relative Humidity (RH), Wind Speed (Ws), Rain, FFMC, DMC, DC, ISI, BUI, and Classes. 

The dataset was split 80/20 for training and testing and scaled using `StandardScaler`. Multiple algorithms were evaluated to find the most robust fit:
* **Linear Regression**
* **Ridge Regression**
* **Lasso Regression**
* **Random Forest Regressor**

### Final Tuned Model
Following an extensive 5-fold `GridSearchCV` testing 108 combinations, the **Random Forest** emerged as the winning model.
* **Optimal Hyperparameters**: `max_depth`: 10, `min_samples_leaf`: 1, `min_samples_split`: 2, `n_estimators`: 100.
* **Test Performance**: Achieved an R² Score of 0.9753, an RMSE of 1.0743, and a MAE of 0.6984.

## Pipeline & Deployment
With the experimentation phase successfully completed, the final model and standard scaler have been exported as `fwi_rf_model.joblib` and `fwi_scaler.joblib`, respectively. The project architecture now moves to the Pipeline Building phase within the `src/` directory to integrate the models into the API backend.
## Quick Start
1. Clone the repo: `git clone <url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the training pipeline: `python -m src.components.Data_Ingestion`