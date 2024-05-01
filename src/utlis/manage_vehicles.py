from pydantic import BaseModel
import csv
import os
from models.vehicle import Vehicle

# Vehicle management class for handling CSV file operations
class VehicleManager:
    def __init__(self, file_path='data/vehicles.csv'):
        self.file_path = file_path
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)  # Ensure the directory exists
        self.isfile_exist = os.path.isfile(self.file_path) and os.path.getsize(self.file_path) > 0
        self.mode = 'a' if self.isfile_exist else 'w'

    def save_vehicles(self, vehicles: list):
        try:
            with open(self.file_path, mode=self.mode, newline='') as file:
                fieldnames = [field for field in Vehicle.model_fields.keys()]
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                if self.mode == 'w':
                    writer.writeheader()
                    self.isfile_exist = True  # Update file existence status after writing header
                for vehicle in vehicles:
                    vehicle.pop('_id', None)
                    # append the vehicle to the csv file
                    writer.writerow(vehicle)
            print(f"Vehicles saved to {self.file_path}")
        except PermissionError:
            print(f"Error: Permission denied when trying to write to {self.file_path}.")
        except IOError as e:
            print(f"IO Error occurred: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def read_vehicles(self):
        vehicles = []
        try:
            with open(self.file_path, mode='r', newline='') as file:
                reader = csv.DictReader(file)
                try:
                    next(reader)  # Attempt to read the first row (after the header)
                except StopIteration:
                    print(f"Error: The file {self.file_path} is empty.")
                    return vehicles  # Return an empty list if the file is empty
                for row in reader:
                    vehicles.append(Vehicle(**row))
            return vehicles
        except FileNotFoundError:
            print(f"Error: The file {self.file_path} does not exist.")
            return vehicles  # Returns an empty list if the file does not exist
        except Exception as e:
            print(f"An unexpected error occurred while reading the file: {e}")
            return vehicles  # Returns whatever could be read before the error occurred

    def delete_vehicles(self):
        try:
            if os.path.exists(self.file_path):
                os.remove(self.file_path)
                print(f"File {self.file_path} deleted")
            else:
                print(f"File {self.file_path} does not exist")
        except PermissionError:
            print(f"Error: Permission denied when trying to delete {self.file_path}.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
