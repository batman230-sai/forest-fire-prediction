import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
#dataset loading
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

if __name__ == "__main__":
    # 1. Initialize the ingestion class
    obj = DataIngestion()
    
    # 2. Call the action method (once you write it)
    # train_data, test_data = obj.initiate_data_ingestion()
    
    print("Data Ingestion setup is executing...")