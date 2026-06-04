import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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

            # Read parameters dynamically during training
            logging.info("Fetching parameters from params.yaml")
            params = read_params()
            model_params = params["model_trainer"]

            # Initialize models, passing the YAML params dynamically using kwargs unpacking
            models = {
                "Linear Regression": LinearRegression(**model_params.get("Linear_Regression", {})),
                "Ridge Regression": Ridge(**model_params.get("Ridge_Regression", {})),
                "Lasso Regression": Lasso(**model_params.get("Lasso_Regression", {})),
                "Random Forest": RandomForestRegressor(**model_params.get("Random_Forest", {}))
            }

            model_report = {}
            
            for name, model in models.items():
                logging.info(f"Training {name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate all metrics (as seen in your notebook)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                logging.info(f"{name} Metrics - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
                
                # Store R2 score to determine the best model
                model_report[name] = r2
            
            # Identify the best performing model based on R2 score
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"Best Base Model: {best_model_name} (R2: {best_model_score:.4f})")

            # Save the winning model
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
            logging.info("Trained model exported successfully.")

            # Return final R2 score of the best model
            final_predictions = best_model.predict(X_test)
            return r2_score(y_test, final_predictions)

        except Exception as e:
            raise CustomException(e, sys)