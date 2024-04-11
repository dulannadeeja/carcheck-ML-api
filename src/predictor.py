import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class CarPricePredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = LinearRegression()
        self.column_value_mappings = {}
        
    def preprocess_data(self):
        cars_data = pd.read_csv(self.data_path)
        
        # Cleanup and prepare data
        cars_data.drop(columns=['Unnamed: 13', 'Unnamed: 14'], inplace=True)
        cars_data.dropna(inplace=True)
        cars_data.drop_duplicates(inplace=True)
        cars_data = cars_data.applymap(lambda s: s.lower() if type(s) == str else s)
        
        # Coverting the all the string values to lower case and strip any leading or trailing spaces
        for col in cars_data.columns:
            if cars_data[col].dtype == object:  # Check if column is of string type
                cars_data[col] = cars_data[col].apply(lambda x: x.lower().strip())
                print(f"Column: {col}")
        
        # Convert string values to numerical codes
        for col in cars_data.columns:
            if cars_data[col].dtype == object:  # Check if column is of string type
                unique_vals = cars_data[col].unique()
                value_to_number = {val: idx for idx, val in enumerate(unique_vals)}
                self.column_value_mappings[col] = value_to_number  # Store the mapping
                cars_data[col] = cars_data[col].map(value_to_number)  # Apply the mapping to convert column values
                
        #shuffle the data
        cars_data = cars_data.sample(frac=1).reset_index(drop=True)
        
        self.input_data = cars_data.drop(columns=['value'])
        self.output_data = cars_data['value']
        
    def train_model(self):
        x_train, x_test, y_train, y_test = train_test_split(self.input_data, self.output_data, test_size=0.2)
        self.model.fit(x_train, y_train)
        # Optionally, store x_test, y_test for evaluation purposes
        
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
        # Make prediction
        input_df = pd.DataFrame([input_features])
        print (input_df)
        return self.model.predict(input_df)
    
    def print_mappings(self):
        for col, mapping in self.column_value_mappings.items():
            print(f"Column '{col}' string to number mapping:")
            for string_val, num_val in mapping.items():
                print(f"  '{string_val}' -> {num_val}")
            print("======================")
            
    def get_mappings(self, column_name, input_value):
        return column_name
        


