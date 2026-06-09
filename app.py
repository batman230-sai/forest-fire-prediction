from fastapi import FastAPI
from pydantic import BaseModel,Field

# Initialize the app
app = FastAPI(title="Forest Fire Prediction API")
# Pydantic Model 
class ModelFeatures(BaseModel):
    """
    Reads input data and validates it automatically.
    """

# GET Request: About Section
@app.get("/about")
def about_me():
    """
    Displays professional information and interactive links.
    """
    Temperature: int=Field(ge=-10,le=60,description="Temperature in Celsius")
    RH: int=Field(ge=10,le=100,description="Relative Humidity in %")
    Ws: int=Field(ge=0,le=100,description="Wind Speed in km/h")
    Rain: int=Field(ge=0,description="Rainfall in mm")
    FFMC: float=Field(ge=0.0,le=100.0,description="Fine Fuel Moisture Code")
    DMC: float=Field(ge=0.0,description="Duff Moisture Code")
    DC: float=Field(ge=0.0,description="Drought Code")
    ISI: float=Field(ge=0.0,description="Initial Spread Index")
    BUI: float=Field(ge=0.0,description="Build Up Index")
    Classes: int=Field(ge=0,le=1)

# POST Request: Prediction Endpoint
@app.post("/predict")
def predict(data: ModelFeatures):
    """
    Collects input data, runs the ML model, and sends back the prediction.
    """