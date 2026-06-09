from fastapi import FastAPI
from pydantic import BaseModel,Field,field_validator, ValidationInfo
from typing import Any
import pickle

with open('artifacts/preprocessor.pkl', 'rb') as file:
    scaler = pickle.load(file)
with open('artifacts/tuned_model.pkl', 'rb') as file:
    model = pickle.load(file)

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
        # Handle Imputation with Mode if they left it blank
        if value in [None, ""]:
            print(f"Classes field empty. Imputing with mode: {DATASET_MODE_CLASS}")
            return DATASET_MODE_CLASS
        if isinstance(value, str):
            clean_value = value.strip().lower()
            if "not" in clean_value:
                return 0
            elif "fire" in clean_value:
                return 1
            else:
                raise ValueError("Classes dropdown value must be 'fire' or 'notfire'.")
                
        return value
# GET Request: About Section
@app.get("/about")
def about_me():
    """
    Displays professional information and interactive links.
    """

# POST Request: Prediction Endpoint
@app.post("/predict")
def predict(data: ModelFeatures):
    """
    Collects input data, runs the ML model, and sends back the prediction.
    """
    raw_data = [list(data.model_dump().values())] 

    scaled_data = scaler.transform(raw_data)
    
    prediction = model.predict(scaled_data)
    
    return {"prediction": float(prediction[0])}