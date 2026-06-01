import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl') 
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_transformer_object(self):
        """Creating scaling pipeline"""
        try:
            numerical_columns=['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'Classes']
            num_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
            preprocessor = ColumnTransformer(transformers=[("num_pipeline", num_pipeline, numerical_columns)])
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    

    