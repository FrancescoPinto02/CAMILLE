from fuzzywuzzy import process
from db.query import get_all_code_smells
import logging

logging.basicConfig(level=logging.INFO)


def code_smell_name_matcher(code_smell_name):
    try:
        # Strip code smell name
        code_smell_name = code_smell_name.strip()

        # Select all the code smells name from DB
        results = get_all_code_smells(["name"])
        choices = [x['name'] for x in results]

        # Return matching results
        match = process.extractOne(code_smell_name, choices)

        return match

    except Exception as e:
        logging.error(f"Error during String Matching: {e}")
        return None
