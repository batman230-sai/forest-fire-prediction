import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

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

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(alpha=1.0),
                "Lasso Regression": Lasso(alpha=0.1),
                "Random Forest": RandomForestRegressor(random_state=42)
            }

            model_report = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                model_report[name] = r2_score(y_test, y_pred)
            
            # Identify the best performing model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"Best Base Model: {best_model_name} (R2: {best_model_score:.4f})")

            # Apply GridSearch if the winner is Random Forest
            if best_model_name == "Random Forest":
                logging.info("Executing GridSearchCV for Random Forest...")
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
                gs = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, n_jobs=-1)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_

            # Save the winning model
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
            logging.info("Trained model exported successfully.")

            final_predictions = best_model.predict(X_test)
            return r2_score(y_test, final_predictions)

        except Exception as e:
            raise CustomException(e, sys)