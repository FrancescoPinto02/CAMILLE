import logging
import pymongo
from db.mongo_db import get_mongo_collection
from utils.string_matcher import code_smell_name_matcher

CONNECTION_ERROR_MESSAGE = "Database connection could not be established"
VALID_FIELDS = ["id", "name", "description", "problems", "solution", "bad_example", "good_example", "type"]


def build_field_projection(fields):
    if not fields:
        return {}
    return {field: 1 for field in fields if field in VALID_FIELDS}


def process_example_fields(code_smell):
    if 'good_example' in code_smell:
        code_smell['good_example'] = code_smell['good_example'].encode().decode('unicode_escape')
    if 'bad_example' in code_smell:
        code_smell['bad_example'] = code_smell['bad_example'].encode().decode('unicode_escape')
    return code_smell


def get_code_smell_by_id(code_smell_id, fields=None):
    mongo_collection, mongo_client = get_mongo_collection()
    projection = build_field_projection(fields)
    try:
        result = mongo_collection.find_one({"id": int(code_smell_id)}, projection)
        if result:
            result = process_example_fields(result)
    except Exception as e:
        logging.error("Error during fetching code smell details: %s", e)
        result = None
    finally:
        mongo_client.close()
    return result


def get_all_code_smells(fields=None):
    mongo_collection, mongo_client = get_mongo_collection()
    projection = build_field_projection(fields)
    try:
        cursor = mongo_collection.find({}, projection).sort("id", pymongo.ASCENDING)
        result = [process_example_fields(doc) for doc in cursor]
    except Exception as e:
        logging.error("Error during fetching all code smells: %s", e)
        result = None
    finally:
        mongo_client.close()
    return result


def get_code_smell_by_name(code_smell_name, fields=None):
    mongo_collection, mongo_client = get_mongo_collection()
    projection = build_field_projection(fields)
    try:
        best_match_name, similarity_score = code_smell_name_matcher(code_smell_name)
        query = {"name": best_match_name} if similarity_score >= 70 else {"name": code_smell_name}
        result = mongo_collection.find_one(query, projection)
        if result:
            result = process_example_fields(result)
    except Exception as e:
        logging.error("Error during fetching code smell details: %s", e)
        result = None
    finally:
        mongo_client.close()
    return result
