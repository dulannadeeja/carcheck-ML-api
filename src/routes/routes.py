from fastapi import APIRouter, HTTPException
from models.vehicle import Vehicle
from config.connection import listing_collection
from schema.schemas import list_serializer, individual_serializer
from bson import ObjectId
from models.vehicle import Vehicle
import pprint
from utlis.manage_ids import VehicleIDManager
from utlis.manage_vehicles import VehicleManager
import pprint
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

router = APIRouter()

printer = pprint.PrettyPrinter(indent=4)

vehicleManager = VehicleManager()
vehicleIdManager = VehicleIDManager()

printer = pprint.PrettyPrinter(indent=4)

@router.post("/api/predict")
async def predict(data: Vehicle):
    try:
        print(data)
        # Convert input to a dictionary for the predictor
        input_features = data.dict()
        # predicted_price = predictor.predict_price(input_features)
        # return {"predicted_price" : predicted_price[0]}
    
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
        
        # # save the vehicles
        vehicleManager.save_vehicles(serialized_vehicles)        
            
        return {"message": "Data loaded successfully"}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 
    
@router.post("/api/load-data")
async def load_data():
    try:
        dataset_path = 'data/car_price_dataset.csv'
        result_path = 'data/car_price_dataset_cleaned.csv'
        df = pd.read_csv(dataset_path)
        
        # Cleanup and prepare data
        
        #drop the last two columns
        df = df.drop(columns=['Unnamed: 13', 'Unnamed: 14'])
        
        # Converting all string columns to lowercase except for the headers
        df = df.map(lambda x: x.lower() if isinstance(x, str) else x)
        
        #list down all the unique values in the columns
        # Looping through each column and printing unique values
        for column in df.columns:
            unique_values = df[column].unique()
            print(f"Unique values in {column}: {unique_values}")
            
        # Replace 'other transmission' with 'automatic' in the 'transmission' column
        df['transmission'] = df['transmission'].replace('other transmission', 'automatic')
        
        # Replace 'used' with 'pre-owned' in the 'condition' column
        df['condition'] = df['condition'].replace('used', 'pre-owned')
        
        # Replace 'station wagon' with 'wagon' in the 'bodyType' column
        df['bodyType'] = df['bodyType'].replace('station wagon', 'wagon')
        
        # Replace 'suv / 4*4' with 'suv' in the 'bodyType' column
        df['bodyType'] = df['bodyType'].replace('suv / 4*4', 'suv')
        
        # Replace 'saloon' with 'sedan' in the 'bodyType' column
        df['bodyType'] = df['bodyType'].replace('saloon', 'sedan')
        
        # Replace all the rows that have vehicleModel as 'prius' to body type 'hatchback'
        df.loc[df['vehicleModel'] == 'prius', 'bodyType'] = 'hatchback'
        
        # Create a dictionary for the new mapping
        color_mapping = {
            'white': 'white', 'pearl white': 'white',
            'gray': 'gray',
            'black': 'black',
            'blue': 'blue', 'light blue': 'blue',
            'red': 'red', 'wine red': 'red', 'dark red': 'red',
            'silver': 'silver',
            'brown': 'brown',
            'orange': 'orange',
            'green': 'green', 'light green': 'green',
            'maroon': 'maroon'
        }

        # Replace the values in the 'exteriorColor' column based on the mapping
        df['exteriorColor'] = df['exteriorColor'].replace(color_mapping)
        
        # Create a dictionary for the mapping
        mapping = {
            1790: 1800, 1700: 1800, 1490: 1500, 1350: 1300, 1791: 1800, 1450: 1500,
            1499: 1500, 1998: 2000, 2400: 2500, 2300: 2500, 2494: 2500, 1190: 1200,
            1197: 1200, 1298: 1300, 2770: 2800, 2750: 2800, 2982: 3000, 2755: 2800,
            2892: 2800, 2780: 2800, 2393: 2500, 2188: 2200
        }
        
        scaler = MinMaxScaler()

        # Replace the values in the 'engineCapacity' column based on the mapping
        df['engineCapacity'] = df['engineCapacity'].replace(mapping)

        # Fitting and transforming the 'value' column
        df['normalized_value'] = scaler.fit_transform(df[['value']])
        
        # Fitting and transforming the 'engineCapacity' column
        df['normalized_engineCapacity'] = scaler.fit_transform(df[['engineCapacity']])
        
        # Fitting and transforming the 'mileage' column
        df['normalized_mileage'] = scaler.fit_transform(df[['mileage']])
        
        # Fitting and transforming the 'manufacturedYear' column
        df['normalized_manufacturedYear'] = scaler.fit_transform(df[['manufacturedYear']])
        
        # Fitting and transforming the 'registeredYear' column
        df['normalized_registeredYear'] = scaler.fit_transform(df[['registeredYear']])

        # Initializing the MinMaxScaler
        scaler = MinMaxScaler()

        # Fitting and transforming the 'value' column
        df['normalized_value'] = scaler.fit_transform(df[['value']])
        
        #save the cleaned dataset
        df.to_csv(result_path, index=False)
        
        # Display the modified DataFrame
        print(df.head())
        
        #normalize the value field
        
        # Sample data creation (you should load your dataset here)
        data = {'value': [9250000, 10475000, 9500000, 5695000, 8825000, 7975000, 6285000]}
        df = pd.DataFrame(data)

        # Initializing the MinMaxScaler
        scaler = MinMaxScaler()

        # Fitting and transforming the 'value' column
        df['normalized_value'] = scaler.fit_transform(df[['value']])
        
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