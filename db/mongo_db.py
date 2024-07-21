import pymongo
from dotenv import load_dotenv
import os
import json

load_dotenv()  # Load environment variables

# Connects to MongoDB
def get_mongo_client():
    return pymongo.MongoClient(os.getenv("MONGO_URI"))


# Retrieves the collection
def get_mongo_collection(client):
    return client[os.getenv("MONGO_DB")]["codesmells"]


def main():
    with get_mongo_client() as client:
        collection = get_mongo_collection(client)
        collection.create_index("id", unique=True)
        print("DB Created.")

        # Read the data from JSON file
        with open('data.json', 'r') as file:
            data = json.load(file)

        # Converts the "id" fields in int
        for item in data:
            if isinstance(item.get('id'), str):
                item['id'] = int(item['id'])

        # Populate
        collection.insert_many(data)
        print("DB Populated.")

        collection.create_index("name")
        print("Indexes created.")


if __name__ == "__main__":
    main()
