import pymongo
from dotenv import load_dotenv
import os
import json

load_dotenv()


def create_mongo_db():
    mongo_client = pymongo.MongoClient(os.getenv("MONGO_URI"))
    mongo_db = mongo_client[os.getenv("MONGO_DB")]
    mongo_collection = mongo_db["codesmells"]
    mongo_collection.create_index("id", unique=True)
    print("DB Created.")
    mongo_client.close()


def populate_mongo_db():
    mongo_client = pymongo.MongoClient(os.getenv("MONGO_URI"))
    mongo_db = mongo_client[os.getenv("MONGO_DB")]
    mongo_collection = mongo_db["codesmells"]

    with open('data.json', 'r') as file:
        data = json.load(file)

    for item in data:
        if isinstance(item.get('id'), str):
            item['id'] = int(item['id'])

    mongo_collection.insert_many(data)
    print("DB Populated.")

    mongo_client.close()


def get_mongo_collection():
    mongo_client = pymongo.MongoClient(os.getenv("MONGO_URI"))
    mongo_db = mongo_client[os.getenv("MONGO_DB")]
    mongo_collection = mongo_db["codesmells"]
    return mongo_collection, mongo_client


def create_indexes():
    mongo_collection, mongo_client = get_mongo_collection()
    try:
        mongo_collection.create_index("id", unique=True)
        mongo_collection.create_index("name")
        print("Indexes created.")
    except Exception as e:
        print(f"Error during creating indexes: {e}")
    finally:
        mongo_client.close()


if __name__ == "__main__":
    create_mongo_db()
    populate_mongo_db()
    create_indexes()

