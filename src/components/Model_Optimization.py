import os
import sys
import numpy as np
from dataclasses import dataclass

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import mlflow
import mlflow.sklearn

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, read_params

@dataclass
class ModelOptimizationConfig:
    
    tuned_model_file_path = os.path.join('artifacts', 'tuned_model.pkl')

class ModelOptimizer:
    def __init__(self):
        self.optimizer_config = ModelOptimizationConfig()

    def initiate_model_optimization(self, train_array, test_array):
        try:
            logging.info("Splitting data for Hyperparameter Tuning")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            
            params = read_params()
            ridge_param_grid = params.get("model_optimization", {}).get("Ridge", {})

            logging.info("Initializing GridSearchCV for Ridge Regression")
            ridge = Ridge()
            
            
            grid_search = GridSearchCV(
                estimator=ridge, 
                param_grid=ridge_param_grid, 
                cv=5, 
                scoring='r2', 
                n_jobs=-1, 
                verbose=2
            )

            
            import os
            os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/batman230-sai/forest-fire-prediction.mlflow"
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
            mlflow.set_experiment("Forest_Fire_Regression_Optimization")

            with mlflow.start_run(run_name="Ridge_Tuned"):
                logging.info("Executing Hyperparameter Tuning. This may take a moment...")
                grid_search.fit(X_train, y_train)

                
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                logging.info(f"Best Parameters found: {best_params}")

                
                y_pred = best_model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                
                mlflow.log_params(best_params)
                mlflow.log_metrics({
                    "Tuned_MAE": mae,
                    "Tuned_RMSE": rmse,
                    "Tuned_R2_Score": r2
                })
                
                
                mlflow.sklearn.log_model(best_model, "tuned_model")
                logging.info(f"Tuned Ridge Model logged to MLflow - R2: {r2:.4f}")

                
                save_object(
                    file_path=self.optimizer_config.tuned_model_file_path, 
                    obj=best_model
                )
                logging.info("Tuned model exported successfully to artifacts/tuned_model.pkl")

                return r2, best_params

        except Exception as e:
            raise CustomException(e, sys)