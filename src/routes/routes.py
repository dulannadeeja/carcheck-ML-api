from fastapi import APIRouter, HTTPException
from config.connection import listing_collection
from schema.schemas import list_serializer
from models.vehicle import InputFeatures
import pprint
from utlis.manage_ids import VehicleIDManager
from utlis.manage_vehicles import VehicleManager
import pprint
import pandas as pd
from predictor import CarPricePredictor

router = APIRouter()

printer = pprint.PrettyPrinter(indent=4)

vehicleManager = VehicleManager()
vehicleIdManager = VehicleIDManager()
predictor = CarPricePredictor('data/car_price_dataset_cleaned.csv')

# define the path to the model
model_path = 'ml_model/model.pkl'
model_mappings_path = 'ml_model/mappings.pkl'

printer = pprint.PrettyPrinter(indent=4)

@router.post("/api/predict")
async def predict(data: InputFeatures):
    try:
        predictor.load_model(model_path)
        predictor.load_mappings(model_mappings_path)
        print(data)
        # Convert input to a dictionary for the predictor
        input_features = data.model_dump()
        predicted_price = predictor.predict_price(input_features)
        return {"predicted_price" : predicted_price}
    
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/api/sync-data")
async def sync_data():
    try:
        #read the data from the database
        vehicles_documents = listing_collection.find()
        
        serialized_vehicles = list_serializer(vehicles_documents)
        
        saved_vehicle_ids = vehicleIdManager.read_vehicle_ids()
        printer.pprint(saved_vehicle_ids)
        
        new_vehicle_ids = []

        # find the new vehicle ids from the serialized data
        for vehicle in serialized_vehicles:
            if vehicle['_id'] not in saved_vehicle_ids:
                new_vehicle_ids.append(vehicle['_id'])
                
        printer.pprint(new_vehicle_ids)
        
        # save the new vehicle ids
        vehicleIdManager.save_vehicle_ids(new_vehicle_ids)
        
        # save the vehicles
        vehicleManager.save_vehicles(serialized_vehicles)       
            
        return {"message": "Data loaded successfully"}
    
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 
    
@router.post("/api/process-data")
async def clean_data():
    try:
        predictor.preprocess_data()
        
        return {"message": "Data loaded successfully"}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 


@router.post("/api/load-external-data")
async def load_external_data():
    try:
        # Load the dataset from the CSV file
        dataset_path = 'data/car_price_dataset.csv'
        
        # Load the dataset
        df = pd.read_csv(dataset_path)
        
        #drop the last two columns
        df = df.drop(columns=['Unnamed: 13', 'Unnamed: 14'])
        
        # Display the first five rows of the DataFrame
        print(df.head())
        
        # make a list from the dataset
        vehicles = df.to_dict('records')
        
        #print the vehicles
        printer.pprint(vehicles)
        
        #save the vehicles
        vehicleManager.save_vehicles(vehicles)
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))    
    

@router.post("/api/train-model")
async def train_model():
    try:
        predictor.preprocess_data()
        predictor.train_model()
        predictor.save_model(model_path)
        predictor.save_mappings(model_mappings_path)
        
        return {"message": "Model trained successfully"}
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
