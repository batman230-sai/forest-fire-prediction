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
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """Creates the scaling pipeline"""
        try:
            numerical_columns = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'Classes']
            
            num_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
            preprocessor = ColumnTransformer(transformers=[("num_pipeline", num_pipeline, numerical_columns)])
            
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def custom_data_cleaning(self, df):
        """Applies exact steps from your Data_Cleaning and Feature_Engineering notebooks"""
        df.columns = df.columns.str.strip()
        df.dropna(subset=['Classes'], inplace=True)
        
        object_columns = df.select_dtypes(include=['object']).columns
        df[object_columns] = df[object_columns].apply(lambda x: x.str.strip())
        
        df['DC'] = pd.to_numeric(df['DC'], errors='coerce')
        df['FWI'] = pd.to_numeric(df['FWI'], errors='coerce')
        df.dropna(inplace=True) 
        
        df.drop(['day', 'month', 'year'], axis=1, inplace=True, errors='ignore')
        df['Classes'] = df['Classes'].map({'fire': 1, 'not fire': 0})
        
        return df

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Applying custom pandas cleaning logic")
            train_df = self.custom_data_cleaning(train_df)
            test_df = self.custom_data_cleaning(test_df)

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "FWI"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying StandardScaler to numerical features")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Concatenate features and target arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the scaler
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj)

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(e, sys)