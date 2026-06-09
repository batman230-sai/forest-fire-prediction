from fastapi import FastAPI
from pydantic import BaseModel,Field,field_validator, ValidationInfo
from typing import Any
import pandas as pd
import joblib
scaler = joblib.load('artifacts/preprocessor.pkl')
model=joblib.load('artifacts/tuned_model.pkl')

DATASET_MEANS = {
    "Temperature": 33,
    "RH": 56,
    "Ws": 15,
    "Rain": 0.6,
    "FFMC": 81.03,
    "DMC": 17.06,
    "DC": 45.67,
    "ISI": 5.83,
    "BUI": 17.96
}
DATASET_MODE_CLASS = 1

# Initialize the app
app = FastAPI(title="Forest Fire Prediction API")
# Pydantic Model 
class ModelFeatures(BaseModel):
    Temperature: int = Field(..., ge=-10, le=60)
    RH: int = Field(..., ge=10, le=100)
    Ws: int = Field(..., ge=0, le=100)
    Rain: float = Field(..., ge=0.0)
    FFMC: float = Field(..., ge=0.0, le=100.0)
    DMC: float = Field(..., ge=0.0)
    DC: float = Field(..., ge=0.0)
    ISI: float = Field(..., ge=0.0)
    BUI: float = Field(..., ge=0.0)
    Classes: int = Field(..., ge=0, le=1)
    @field_validator('Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', mode='before')
    @classmethod
    def impute_numerical_fields(cls, value: Any, info: ValidationInfo) -> Any:
        field_name = info.field_name  # Identifies which field is being processed
        if value in [None, "", 0, 0.0, "0", "0.0"]:
            print(f"Imputing missing/zero value for {field_name} with mean: {DATASET_MEANS[field_name]}")
            return DATASET_MEANS[field_name]
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                raise ValueError(f"{field_name} must be a valid numeric value.")
                
        return value
    @field_validator('Classes', mode='before')
    @classmethod
    def convert_and_impute_classes(cls, value: Any) -> int:
        if value in [None, ""]:
            print(f"Classes field empty. Imputing with mode: {DATASET_MODE_CLASS}")
            return DATASET_MODE_CLASS
        
        if isinstance(value, str):
            clean_value = value.strip().lower()
            if clean_value in ["not fire", "notfire", "not_fire", "0"]:  # ✅ explicit
                return 0
            elif clean_value in ["fire", "1"]:                           # ✅ explicit
                return 1
            else:
                raise ValueError("Classes must be 'fire' or 'not fire'.")
        
        if isinstance(value, int) and value in [0, 1]:
            return value
        
        raise ValueError("Classes must be 0 or 1.")
# GET Request: About Section
@app.get("/about")
def about_me():
    """
    Displays professional information and interactive links.
    """
    return {
        "developer": "Durgavajula Sai Kumar",
        "role": "Data Science & Machine Learning | Incoming GET at LTI Mindtree",
        "project": "Forest Fire Prediction System",
        "connect": {
            "github": "https://github.com/batman230-sai",
            "linkedin": "https://www.linkedin.com/in/sai-kumar-474689289"
        }
    }

# POST Request: Prediction Endpoint
@app.post("/predict")
def predict(data: ModelFeatures):
    """
    Collects input data, runs the ML model, and sends back the prediction.
    """
    raw_data = pd.DataFrame([data.model_dump()])
    print(f"Classes received: {data.Classes}")
    scaled_data = scaler.transform(raw_data)
    
    prediction = model.predict(scaled_data)
    
    return {"prediction": float(prediction[0])}