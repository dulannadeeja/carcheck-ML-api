from fastapi import HTTPException
from fastapi import FastAPI
import pickle
import pandas as pd  # Import pandas
from predictor import CarPricePredictor
from fastapi.middleware.cors import CORSMiddleware
from routes.routes import router

app = FastAPI()

# connect router
app.include_router(router)
 
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
