
from dotenv import load_dotenv, find_dotenv
import os
import pprint
from pymongo import MongoClient

load_dotenv(find_dotenv())

# Load the environment variables for the MongoDB connection
password = os.environ.get("MONGO_PASSWORD")
username = os.environ.get("MONGO_USERNAME")
cluster = os.environ.get("MONGO_CLUSTER")
uri = f"""mongodb+srv://{username}:{password}@{cluster}/node_project?retryWrites=true&w=majority"""

# connect to the MongoDB cluster
client = MongoClient(uri)
database = client['node_project']
listing_collection = database['listings']

print("Connected to MongoDB")

