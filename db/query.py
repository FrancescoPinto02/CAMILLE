import logging
import os

import pymongo
from db.mongo_db import get_mongo_client  # Adjusted import statement

# Constants for database operations
CONNECTION_ERROR_MESSAGE = "Database connection could not be established"
VALID_FIELDS = ["id", "name", "description", "problems", "solution",
                "bad_example", "good_example", "prompt_example", "type"]


# Create a MongoDB projection dictionary based on the specified fields.
def build_field_projection(fields):
    projection = {}

    # Check if provided fields are valid
    if fields:
        for field in fields:
            if field in VALID_FIELDS:
                projection[field] = 1  # Include this field in the projection

    return projection


# Decode escape sequences in code examples
def process_example_fields(code_smell):
    for key in ['good_example', 'bad_example', 'prompt_example']:
        if key in code_smell:
            code_smell[key] = code_smell[key].encode().decode('unicode_escape')
    return code_smell


def get_mongo_collection():
    # Retrieve the 'codesmells' collection from the MongoDB
    client = get_mongo_client()
    return client[os.getenv("MONGO_DB")]["codesmells"], client


def get_code_smell_by_id(code_smell_id, fields=None):
    # Fetch a code smell by its ID from the database
    mongo_collection, mongo_client = get_mongo_collection()
    projection = build_field_projection(fields)
    try:
        result = mongo_collection.find_one({"id": int(code_smell_id)}, projection)
        if result:
            result = process_example_fields(result)
    except Exception as e:
        logging.error(f"Error fetching code smell details: {e}")
        result = None
    finally:
        mongo_client.close()
    return result


def get_all_code_smells(fields=None):
    # Fetch all code smells from the database, sorted by 'id'
    mongo_collection, mongo_client = get_mongo_collection()
    projection = build_field_projection(fields)
    try:
        cursor = mongo_collection.find({}, projection).sort("id", pymongo.ASCENDING)
        result = [process_example_fields(doc) for doc in cursor]
    except Exception as e:
        logging.error(f"Error fetching all code smells: {e}")
        result = None
    finally:
        mongo_client.close()
    return result


# Fetch a code smell by its name
def get_code_smell_by_name(code_smell_name, fields=None):

    from utils.string_matcher import code_smell_name_matcher
    mongo_collection, mongo_client = get_mongo_collection()
    projection = build_field_projection(fields)
    try:
        # Find Best Match
        best_match_name, similarity_score = code_smell_name_matcher(code_smell_name)
        query = {"name": best_match_name} if similarity_score >= 70 else {"name": code_smell_name}
        result = mongo_collection.find_one(query, projection)
        if result:
            result = process_example_fields(result)
    except Exception as e:
        logging.error(f"Error fetching code smell details: {e}")
        result = None
    finally:
        mongo_client.close()
    return result
