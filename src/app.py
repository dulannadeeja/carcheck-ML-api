from fastapi import HTTPException
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd  # Import pandas
from predictor import CarPricePredictor
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
 
# define the path to the model
model_path = 'model/model.pkl'
model_mappings_path = 'model/mappings.pkl'

# Initialize the predictor
predictor = CarPricePredictor('model/car_price_dataset.csv')

# check if the model exists
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    predictor.load_model(model_path)
    predictor.load_mappings(model_mappings_path)
except FileNotFoundError:
    predictor.preprocess_data()
    predictor.train_model()
    predictor.save_model(model_path)
    predictor.save_mappings(model_mappings_path)

# Print the mappings for each column in the dataset
predictor.print_mappings()

# Add CORSMiddleware to the application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class ModelInput(BaseModel):
    make: str
    model: str
    manufacturedYear: int
    registeredYear: int
    mileage: int
    previousOwners: int
    exteriorColor: str
    fuelType: str
    condition: str
    transmission: str
    bodyType: str
    engineCapacity: int

@app.post("/api/predict")
async def predict(data: ModelInput):
    try:
        print(data)
        # Convert input to a dictionary for the predictor
        input_features = data.dict()
        predicted_price = predictor.predict_price(input_features)
        return {"predicted_price" : predicted_price[0]}
    
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
