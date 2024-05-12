from fastapi import APIRouter, HTTPException
from config.connection import listing_collection, model_info_collection
from schema.schemas import list_serializer
from models.vehicle import InputFeatures
import pprint
from utlis.manage_ids import VehicleIDManager
from utlis.manage_vehicles import VehicleManager
import pprint
import pandas as pd
from predictor import CarPricePredictor

# define the paths
model_path = 'ml_model/model.pkl'
model_mappings_path = 'ml_model/mappings.pkl'
sample_data_path = 'data/car_price_dataset.csv'
cleaned_data_path = 'data/car_price_dataset_cleaned.csv'
vehicles_path = 'data/vehicles.csv'
saved_vehicle_ids_path = 'data/saved_vehicle_ids.csv'
feature_scaler_path = 'ml_model/features_scaler.joblib'
value_scaler_path = 'ml_model/value_scaler.joblib'

router = APIRouter()

printer = pprint.PrettyPrinter(indent=4)

vehicleManager = VehicleManager(vehicles_path)
vehicleIdManager = VehicleIDManager(saved_vehicle_ids_path)
predictor = CarPricePredictor(
    sample_data_path,
    cleaned_data_path,
    model_mappings_path,
    model_path,
    feature_scaler_path,
    value_scaler_path,
    vehicles_path,
    saved_vehicle_ids_path,
)



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
    

@router.put("/api/sync-data")
async def sync_data():
    try:
        printer.pprint("Syncing data")
        #read the data from the database
        vehicles_documents = listing_collection.find({"status": {"$ne": "Draft"}})
        
        # filter the required fields from the cursor
        expected_fields = ['_id', 'make', 'vehicleModel','manufacturedYear','registeredYear','mileage','numberOfPreviousOwners','exteriorColor','fuelType','condition','transmission','bodyType','engineCapacity','currentPrice']  # Define your expected field names
        filtered_vehicles = []
        for vehicle in vehicles_documents:
            filtered_vehicle = {key: vehicle[key] for key in expected_fields if key in vehicle}
            filtered_vehicles.append(filtered_vehicle)  
        
       # serialize the vehicles
        serialized_vehicles = list_serializer(filtered_vehicles)
        
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
    
@router.get("/api/process-data")
async def clean_data():
    try:
        predictor.preprocess_data()
        
        return {"message": "Data loaded successfully"}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 


@router.get("/api/load-initial-data")
async def load_external_data():
    try:
        # Load the dataset
        df = pd.read_csv(sample_data_path)
        
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
        
        return {"message": "Initial data loaded successfully"}
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))    
    

@router.get("/api/train-model")
async def train_model():
    try:
        predictor.preprocess_data()
        result = predictor.train_model()
        predictor.save_model(model_path)
        predictor.save_mappings(model_mappings_path)
        
        r_Squared = result['r_squared']
        num_rows = result['num_rows']
        
        return {
            "message": "Model trained successfully",
            "r_squared": r_Squared,
            "num_rows": num_rows
            }
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@router.delete("/api/clean")
async def clean():
    try:
        predictor.clean_model()
        return {"message": "Model cleaned successfully"}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/api/model-info")
async def model_info():
    try:
        #find the latest model info doc from the model_info collection
        model_info = model_info_collection.find_one(sort=[("operationDate", -1)])
        if model_info is None:
            return {"message": "No model info available"}
        
        return {
            'message': "Model info retrieved successfully",
            'opearationDate': model_info['operationDate'],
            'version': model_info['version'],
            'accuracy': model_info['accuracy'],
            'totalRecords': model_info['totalRecords']
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))    