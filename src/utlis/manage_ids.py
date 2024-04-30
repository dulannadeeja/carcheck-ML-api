import csv
import os

class VehicleIDManager:
    def __init__(self, file_path='data/saved_vehicle_ids.csv'):
        self.file_path = file_path
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self.isfile_exist = os.path.isfile(self.file_path) and os.path.getsize(self.file_path) > 0
        self.mode = 'a' if self.isfile_exist else 'w'

    def save_vehicle_ids(self, vehicle_ids: list):
        try:
            with open(self.file_path, mode=self.mode, newline='') as file:
                writer = csv.writer(file)
                # Only write the header if the file does not exist yet
                if not self.isfile_exist:
                    writer.writerow(['saved_id'])
                    self.isfile_exist = True  # Update the file existence status after writing the header
                for vehicle_id in vehicle_ids:
                    writer.writerow([vehicle_id])
            print(f"Vehicle IDs saved to {self.file_path}")
        except PermissionError:
            print(f"Error: Permission denied when trying to write to {self.file_path}.")
        except IOError as e:
            print(f"IO Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def read_vehicle_ids(self):
        vehicle_ids = []
        try:
            with open(self.file_path, mode='r', newline='') as file:
                reader = csv.reader(file)
                next(reader)  # Attempt to skip the header, can raise StopIteration if file is empty
                for row in reader:
                    vehicle_ids.append(row[0])
        except FileNotFoundError:
            print(f"Error: The file {self.file_path} does not exist.")
        except StopIteration:
            print("Error: The file {self.file_path} is empty.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        else:
            print(f"Vehicle IDs read from {self.file_path}")
        finally:
            return vehicle_ids


    def delete_vehicle_ids(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
            print(f"File {self.file_path} deleted")
        else:
            print(f"File {self.file_path} does not exist")
