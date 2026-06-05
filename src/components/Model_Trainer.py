import os
import sys
import numpy as np
from dataclasses import dataclass

# Import all algorithms
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# MLflow tracking
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, read_params

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting transformed data into X and y")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            params = read_params()
            model_params = params["model_trainer"]

            # Initialize all 6 models dynamically
            models = {
                "Linear Regression": LinearRegression(**model_params.get("Linear_Regression", {})),
                "Ridge Regression": Ridge(**model_params.get("Ridge_Regression", {})),
                "Lasso Regression": Lasso(**model_params.get("Lasso_Regression", {})),
                "SVR": SVR(**model_params.get("SVR", {})),
                "Random Forest": RandomForestRegressor(**model_params.get("Random_Forest", {})),
                "XGBoost": XGBRegressor(**model_params.get("XGBoost", {}))
            }

            model_report = {}

            # Remove this:
# mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Replace with your DagsHub remote URI:

            os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/batman230-sai/forest-fire-prediction.mlflow"
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
            # Name your experiment
            mlflow.set_experiment("Forest_Fire_Regression_Experiments")

            for name, model in models.items():
                logging.info(f"Training {name}...")
                
                # Start an MLflow run for each model
                with mlflow.start_run(run_name=name):
                    
                    # 1. Train the model
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # 2. Calculate Evaluation Metrics
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
                    
                    # 3. Log Parameters and Metrics to MLflow
                    # Fetch the specific parameters for this model from the yaml dict
                    current_model_params = model_params.get(name.replace(" ", "_"), {})
                    if current_model_params:
                        mlflow.log_params(current_model_params)
                    
                    mlflow.log_metrics({
                        "MAE": mae,
                        "MSE": mse,
                        "RMSE": rmse,
                        "R2_Score": r2
                    })
                    
                    # 4. Log the model artifact to MLflow
                    mlflow.sklearn.log_model(model, "model")
                    
                    logging.info(f"{name} logged to MLflow - R2: {r2:.4f}")
                    
                    # Store R2 score to determine the overall best model to save locally
                    model_report[name] = r2
            
            # Identify the best performing model based on R2 score
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"Best Base Model: {best_model_name} (R2: {best_model_score:.4f})")

            # Save the winning model locally for your pipeline
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
            logging.info("Best trained model exported successfully to artifacts.")

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)