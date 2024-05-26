from fuzzywuzzy import process
from db.db_pool import get_connection
import logging

logging.basicConfig(level=logging.INFO)


def code_smell_name_matcher(code_smell_name):
    connection = get_connection()
    try:
        cursor = connection.cursor()

        # Strip code smell name
        code_smell_name = code_smell_name.strip()

        # Select all the choices from the DB
        query = "SELECT name FROM codesmell"
        cursor.execute(query)
        result = cursor.fetchall()
        choices = [row[0] for row in result]
        cursor.close()

        # Return matching results
        return process.extractOne(code_smell_name, choices)

    except Exception as e:
        logging.error("Error during code smell name matching: %s", e)

    finally:
        if connection:
            connection.close()
