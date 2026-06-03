import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, read_params # Imported read_params here

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
            rf_params = params["model_trainer"]

            # Initialize models, passing the YAML params to Random Forest
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(alpha=1.0),
                "Lasso Regression": Lasso(alpha=0.1),
                "Random Forest": RandomForestRegressor(
                    n_estimators=rf_params["n_estimators"],
                    max_depth=rf_params["max_depth"],
                    min_samples_split=rf_params["min_samples_split"],
                    random_state=42
                )
            }

            model_report = {}
            for name, model in models.items():
                logging.info(f"Training {name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                model_report[name] = r2_score(y_test, y_pred)
            
            # Identify the best performing model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"Best Base Model: {best_model_name} (R2: {best_model_score:.4f})")

            # Save the winning model
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
            logging.info("Trained model exported successfully.")

            final_predictions = best_model.predict(X_test)
            return r2_score(y_test, final_predictions)

        except Exception as e:
            raise CustomException(e, sys)