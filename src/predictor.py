import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from joblib import load
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
import os
import logging
from sklearn.metrics import r2_score
from models.modelInfo import ModelInfo
import datetime
from config.connection import model_info_collection

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CarPricePredictor:
    def __init__(self, 
                 sample_data_path:str, 
                 clean_data_path:str,
                 mappings_path:str,
                 model_path:str,
                 feature_scaler_path:str,
                 value_scaler_path:str,
                 vehicles_path:str,
                 saved_vehicle_ids_path:str
                 ):
        self.sample_data_path = sample_data_path
        self.clean_data_path = clean_data_path
        self.mappings_path = mappings_path
        self.model_path = model_path
        self.vehicle_path = vehicles_path
        self.saved_vehicle_ids_path = saved_vehicle_ids_path
        self.feature_scaler_path = feature_scaler_path
        self.value_scaler_path = value_scaler_path
        self.model = LinearRegression()
        self.column_value_mappings = {}
        
     # Cleanup and prepare data for analysis    
    
    def preprocess_data(self):
        # Load the dataset
        df = pd.read_csv(self.vehicle_path)
        
        print(df.head())
        
        # Converting all string columns to lowercase except for the headers
        df = df.map(lambda x: x.lower() if isinstance(x, str) else x)
        
        #drop duplicates and missing values
        df.dropna()
        df.drop_duplicates()
        
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
        
        # Set the body type to 'sedan' for all 'yaris' models
        df.loc[df['vehicleModel'] == 'yaris', 'bodyType'] = 'sedan'
        
        # Set the body type to 'hatchback' for all 'vitz' models
        df.loc[df['vehicleModel'] == 'vitz', 'bodyType'] = 'hatchback'
        
        # Set the body type to 'suv' for all 'chr' models
        df.loc[df['vehicleModel'] == 'chr', 'bodyType'] = 'suv'
        
        # set the body type to 'sedan' for all 'axio' models
        df.loc[df['vehicleModel'] == 'axio', 'bodyType'] = 'sedan'
        
        # set the body type to 'pickup' for all 'hilux' models
        df.loc[df['vehicleModel'] == 'hilux', 'bodyType'] = 'pickup'
        
        # set the body type to 'hatcback' for all 'aqua' models
        df.loc[df['vehicleModel'] == 'aqua', 'bodyType'] = 'hatchback'
        
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
        # Replace the values in the 'engineCapacity' column based on the mapping
        df['engineCapacity'] = df['engineCapacity'].replace(mapping)
        
        
        # Convert string values to numerical codes
        for col in df.columns:
            if df[col].dtype == object:  # Check if column is of string type
                unique_vals = df[col].unique()
                value_to_number = {val: idx for idx, val in enumerate(unique_vals)}
                self.column_value_mappings[col] = value_to_number  # Store the mapping
                df[col] = df[col].map(value_to_number)  # Apply the mapping to convert column values
        
        
        # scale the dataframe values
        features_scaler = MinMaxScaler()
        features_df = df.drop(columns=['value'])
        features_df = pd.DataFrame(features_scaler.fit_transform(features_df), columns=features_df.columns)
        dump(features_scaler, self.feature_scaler_path)
        
        #overwrite the original dataframe with the scaled values
        df = pd.concat([features_df, df['value']], axis=1)
        
        value_scaler = MinMaxScaler()
        df['value'] = value_scaler.fit_transform(df[['value']])
        dump(value_scaler, self.value_scaler_path)
        
        #save the cleaned dataset
        df.to_csv(self.clean_data_path, index=False)
        
        # Display the modified DataFrame
        print(df.head())
             
        #shuffle the data
        cars_data = df.sample(frac=1).reset_index(drop=True)
        
        self.input_data = cars_data.drop(columns=['value'])
        self.output_data = cars_data['value']
        
    def train_model(self):
        x_train, x_test, y_train, y_test = train_test_split(self.input_data, self.output_data, test_size=0.2)
        self.model.fit(x_train, y_train)
        # evaluate the model
        # Predict the target variable for the test data
        y_pred = self.model.predict(x_test)

        # Calculate the R-squared value
        r_squared = r2_score(y_test, y_pred)

        # Convert R-squared to a percentage
        r_squared_percentage = np.round(r_squared * 100, 2)
        
        #count the number of rows in the dataset
        num_rows = len(self.input_data)
        
        #save model info to the database
        self.save_model_info(r_squared_percentage, num_rows)
        
        return {
            "r_squared": r_squared_percentage,
            "num_rows": num_rows
        }
        
    def load_model(self, model_path):
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)
    
    def load_mappings(self, mappings_path):
        with open(mappings_path, 'rb') as file:
            self.column_value_mappings = pickle.load(file)
            
    def save_mappings(self, mappings_path):
        with open(mappings_path, 'wb') as file:
            pickle.dump(self.column_value_mappings, file)
        
    def save_model(self, model_path):
        with open(model_path, 'wb') as file:
            pickle.dump(self.model, file)
        
    def predict_price(self, input_features):
        print (input_features)
        
        # get the column names from the input features where the value is a string
        inValCols = [col for col in input_features if isinstance(input_features[col], str)]
        print(f"inValCols: {inValCols}")
        
        # get the mapped numerical values for the string values
        for col in inValCols:
            input_features[col] = self.column_value_mappings[col][input_features[col].lower().strip()]
        print (input_features)
        
         # Create a DataFrame for transformation
        input_df = pd.DataFrame([input_features])
        
        # normalize the input features
        # input features does not have the value column
        features_scaler = load(self.feature_scaler_path)
        input_df = pd.DataFrame(features_scaler.transform(input_df), columns=input_df.columns)
        print (input_df)
        
        predicted_value = self.model.predict(input_df)
        
        normalized_model_output = np.array([[predicted_value]])

        # Reshape the output to match the expected input shape of the scaler
        normalized_model_output = normalized_model_output.reshape(-1, 1)

        value_scaler = load(self.value_scaler_path)
        # Denormalize the output using the inverse transform of the scaler
        denormalized_output = value_scaler.inverse_transform(normalized_model_output)
    
        print (denormalized_output)
        return denormalized_output[0][0]
        
    def print_mappings(self):
        for col, mapping in self.column_value_mappings.items():
            print(f"Column '{col}' string to number mapping:")
            for string_val, num_val in mapping.items():
                print(f"  '{string_val}' -> {num_val}")
            print("======================")
            
    def get_mappings(self, column_name):
        return column_name
    
    def clean_model(self,):
        # Clean up the model and resources
        self.model = LinearRegression()
        self.column_value_mappings = {}
        
        # List of paths to delete
        paths_to_delete = [
            self.clean_data_path, 
            self.mappings_path,
            self.model_path, 
            self.feature_scaler_path, 
            self.value_scaler_path, 
            self.vehicle_path, 
            self.saved_vehicle_ids_path
        ]
        
        # Delete the saved files
        for path in paths_to_delete:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logging.info(f"Successfully removed {path}")
                else:
                    logging.warning(f"No file found to remove at: {path}")
            except Exception as e:
                logging.error(f"Failed to remove {path}. Error: {e}")
        
    def save_model_info(self, accuracy:float, num_rows:int):
        current_datetime = datetime.datetime.now()
        model_info:ModelInfo = ModelInfo(
            operationDate=current_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            version='1.0.0',
            accuracy=accuracy,
            totalRecords=num_rows
            )
        record = model_info.model_dump()
        model_info_collection.insert_one(record)
        print("Model info saved successfully")

