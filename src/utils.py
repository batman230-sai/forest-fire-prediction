import os
import sys
import joblib
import yaml
from src.exception import CustomException

def read_params(config_path="params.yaml"):
    """Reads the params.yaml file and returns the configuration."""
    try:
        with open(config_path) as yaml_file:
            config = yaml.safe_load(yaml_file)
        return config
    except Exception as e:
        raise CustomException(e, sys)

def save_object(file_path, obj):
    """Saves a python object (like a model or scaler) to a specified file path."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)