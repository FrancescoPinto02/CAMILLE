import logging

from db.db_pool import get_connection
from utils.string_matcher import code_smell_name_matcher

CONNECTION_ERROR_MESSAGE = "Database connection could not be established"
VALID_FIELDS = ["id", "name", "description", "problems", "solution", "bad_example", "good_example", "type"]


def get_code_smells_by_id(code_smell_id, fields=None):
    fields_str = build_field_string(fields)
    query = f"SELECT {fields_str} FROM codesmell WHERE id=%s"
    result = None
    connection = get_connection()

    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute(query, (code_smell_id,))
            result = cursor.fetchone()
            cursor.close()
        except Exception as e:
            logging.error("Error during fetching code smell details: %s", e)
        finally:
            connection.close()
    else:
        logging.error("Error during getCodeSmellByID: %s", CONNECTION_ERROR_MESSAGE)

    return result


def get_all_code_smells(fields=None):
    fields_str = build_field_string(fields)
    query = f"SELECT {fields_str} FROM codesmell ORDER BY id ASC"
    result = None
    connection = get_connection()

    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute(query)
            result = cursor.fetchall()
            cursor.close()
        except Exception as e:
            logging.error("Error during fetching all codesmells: %s", e)
        finally:
            connection.close()
    else:
        logging.error("Error during getAllCodeSmells: %s", CONNECTION_ERROR_MESSAGE)

    return result


def get_code_smells_by_name(code_smell_name, fields=None):
    fields_str = build_field_string(fields)
    query = f"SELECT {fields_str} FROM codesmell WHERE name=%s"
    result = None
    connection = get_connection()

    if connection:
        try:
            cursor = connection.cursor(dictionary=True)

            best_match_name, similarity_score = code_smell_name_matcher(code_smell_name)
            if similarity_score >= 70:
                cursor.execute(query, (best_match_name,))
            else:
                cursor.execute(query, (code_smell_name,))

            result = cursor.fetchone()
            cursor.close()
        except Exception as e:
            logging.error("Error during fetching code smell details: %s", e)
        finally:
            connection.close()
    else:
        logging.error("Error during getCodeSmellByName: %s", CONNECTION_ERROR_MESSAGE)

    return result


def build_field_string(fields):
    if not fields:
        return "*"

    valid_fields = []
    for field in fields:
        if field in VALID_FIELDS:
            valid_fields.append(field)

    if not valid_fields:
        return "*"

    return ", ".join(valid_fields)
