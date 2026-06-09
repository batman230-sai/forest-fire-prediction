from fastapi import FastAPI
from pydantic import BaseModel,Field,field_validator, StrictInt, StrictFloat

DATASET_MEANS = {
    "Temperature": 32,
    "RH": 62,
    "Ws": 16,
    "Rain": 1.0,
    "FFMC": 78.0,
    "DMC": 15.0,
    "DC": 50.0,
    "ISI": 5.0,
    "BUI": 17.0
}
DATASET_MODE_CLASS = 1

# Initialize the app
app = FastAPI(title="Forest Fire Prediction API")
# Pydantic Model 
class ModelFeatures(BaseModel):
    """
    Reads input data and validates it automatically.
    """
    Temperature: int=Field(ge=-10,le=60,description="Temperature in Celsius")
    RH: StrictInt=Field(ge=10,le=100,description="Relative Humidity in %")
    Ws: StrictInt=Field(ge=0,le=100,description="Wind Speed in km/h")
    Rain: StrictFloat=Field(ge=0,description="Rainfall in mm")
    FFMC: StrictFloat=Field(ge=0.0,le=100.0,description="Fine Fuel Moisture Code")
    DMC: StrictFloat=Field(ge=0.0,description="Duff Moisture Code")
    DC: StrictFloat=Field(ge=0.0,description="Drought Code")
    ISI: StrictFloat=Field(ge=0.0,description="Initial Spread Index")
    BUI: StrictFloat=Field(ge=0.0,description="Build Up Index")
    Classes: StrictInt=Field(ge=0,le=1)

    @field_validator('Temperature')
    @classmethod
    def check_realistic_temp(cls, value: int) -> int:
        """Example of a custom validator for logical checks."""
        if value > 50:
            # You might want to log this in a real MLOps pipeline
            print(f"Warning: Unusually high temperature recorded ({value}°C)")
        return value

    @field_validator('RH')
    @classmethod
    def check_humidity(cls, value: int) -> int:
        """Ensure relative humidity makes physical sense."""
        if value < 15:
            print("Warning: Extremely dry conditions reported.")
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