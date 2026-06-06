import sys
from src.logger import logging
from src.exception import CustomException

# Import the components you already built
from src.components.Data_Ingestion import DataIngestion
from src.components.Data_Transformation import DataTransformation
from src.components.Model_Trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("Starting the Training Pipeline Orchestration...")

            # Step 1: Data Ingestion
            logging.info("--- Step 1: Initiating Data Ingestion ---")
            ingestion = DataIngestion()
            train_data_path, test_data_path = ingestion.initiate_data_ingestion()
            
            # Step 2: Data Transformation
            logging.info("--- Step 2: Initiating Data Transformation ---")
            transformation = DataTransformation()
            train_arr, test_arr, _ = transformation.initiate_data_transformation(
                train_data_path, 
                test_data_path
            )
            
            # Step 3: Model Training
            logging.info("--- Step 3: Initiating Model Training ---")
            trainer = ModelTrainer()
            # This should train the model, evaluate it, and save tuned_model.pkl
            r2_score = trainer.initiate_model_trainer(train_arr, test_arr)
            
            logging.info(f"Training Pipeline Completed Successfully. Final Model R2 Score: {r2_score}")
            print(f"✅ Pipeline complete! Model trained with R2 Score: {r2_score}")

        except Exception as e:
            logging.error("Exception occurred in the Training Pipeline")
            raise CustomException(e, sys)

# This allows you to run the training pipeline directly from the terminal
if __name__ == "__main__":
    try:
        pipeline = TrainPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        print(f"Pipeline failed: {e}")