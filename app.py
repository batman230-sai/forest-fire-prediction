from fastapi import FastAPI
from pydantic import BaseModel

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

# POST Request: Prediction Endpoint
# ---------------------------------------------------------
@app.post("/predict")
def predict(data: ModelFeatures):
    """
    Collects input data, runs the ML model, and sends back the prediction.
    """