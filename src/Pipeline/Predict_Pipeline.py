import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        # Optimization: Load artifacts once during initialization to reduce prediction latency
        try:
            model_path = os.path.join("artifacts", "tuned_model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            print("Loading artifacts...")
            self.model = load_object(file_path=model_path)
            self.preprocessor = load_object(file_path=preprocessor_path)
            print("Artifacts loaded successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features):
        try:
            # Artifacts are already in memory, simply transform and predict
            data_scaled = self.preprocessor.transform(features)
            preds = self.model.predict(data_scaled)
            
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 Temperature: int,
                 RH: int,
                 Ws: int,
                 Rain: float,
                 FFMC: float,
                 DMC: float,
                 DC: float,
                 ISI: float,
                 BUI: float,
                 Classes: str):
        
        self.Temperature = Temperature
        self.RH = RH
        self.Ws = Ws
        self.Rain = Rain
        self.FFMC = FFMC
        self.DMC = DMC
        self.DC = DC
        self.ISI = ISI
        self.BUI = BUI
        self.Classes = Classes

    def get_data_as_df(self):
        try:
            # Map the incoming string to the integer the preprocessor expects
            class_mapping = {'fire': 1, 'not fire': 0}
            
            # Safe conversion: ensures it won't crash if an int/float is passed by mistake
            class_val = str(self.Classes).strip().lower()
            mapped_class = class_mapping.get(class_val, 0) # Defaults to 0 if unexpected string

            custom_data_dict = {
                "Temperature": [self.Temperature],
                "RH": [self.RH],
                "Ws": [self.Ws],
                "Rain": [self.Rain],
                "FFMC": [self.FFMC],
                "DMC": [self.DMC],
                "DC": [self.DC],
                "ISI": [self.ISI],
                "BUI": [self.BUI],
                "Classes": [mapped_class]
            }
            
            return pd.DataFrame(custom_data_dict)

        except Exception as e:
            raise CustomException(e, sys)