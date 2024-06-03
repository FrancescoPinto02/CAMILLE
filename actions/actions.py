# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

from typing import Any, Text, Dict, List

import mysql.connector
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from db.db_pool import get_connection
import logging
from utils.string_matcher import code_smell_name_matcher
import requests
from dotenv import load_dotenv
import os
import json

logging.basicConfig(level=logging.INFO)

ERROR_MESSAGE = "Sorry, there was a problem... Please try again."
CONNECTION_ERROR_MASSAGE = "Database connection could not be established"

load_dotenv()


class ActionDefaultFallback(Action):

    def name(self) -> Text:
        return "action_default_fallback"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(template="utter_default")
        return []


# REPLY WITH THE LIST OF CODE SMELLS RETRIEVED FROM DB
class ActionGetCodeSmellsList(Action):

    def name(self) -> Text:
        return "action_get_code_smells_list"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Connecting to DB
        connection = get_connection()

        if connection:
            try:
                cursor = connection.cursor()

                # Execute Query
                query = "SELECT id, name FROM codesmell ORDER BY id"
                cursor.execute(query)
                result = cursor.fetchall()

                if result:
                    message = "Here is the list of code smells:\n"
                    for id, name in result:
                        message += f"{id}: {name}\n"
                    dispatcher.utter_message(text=message)
                else:
                    dispatcher.utter_message(text="No code smells found in the database.")

                cursor.close()

            except Exception as e:
                dispatcher.utter_message(text=ERROR_MESSAGE)
                logging.error("Error during action code smell list: %s", e)
            finally:
                connection.close()
        else:
            dispatcher.utter_message(text=ERROR_MESSAGE)
            logging.error("Error during action code smells list: %s", CONNECTION_ERROR_MASSAGE)

        return []


class ActionProvideCodeSmellDetails(Action):

    def name(self) -> Text:
        return "action_provide_code_smell_details"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        code_smell_id = next(tracker.get_latest_entity_values("code_smell_id"), None)
        code_smell_name = next(tracker.get_latest_entity_values("code_smell_name"), None)

        connection = get_connection()
        if connection:
            try:
                cursor = connection.cursor(dictionary=True)

                if code_smell_id:
                    query = "SELECT description FROM codesmell WHERE id=%s"
                    cursor.execute(query, (code_smell_id,))
                    result = cursor.fetchone()
                elif code_smell_name:
                    query = "SELECT description FROM codesmell WHERE name=%s"
                    best_match_name, similarity_score = code_smell_name_matcher(code_smell_name)
                    if similarity_score >= 70:
                        cursor.execute(query, (best_match_name,))
                    else:
                        cursor.execute(query, (code_smell_name,))
                    result = cursor.fetchone()
                else:
                    dispatcher.utter_message(
                        text="I'm sorry, I didn't understand what code smell you were referring to.")
                    return []

                if result:
                    code_smell_details = result["description"]
                    dispatcher.utter_message(text=f"{code_smell_details}")
                else:
                    dispatcher.utter_message(text="Sorry, I couldn't find any details about that.")

                cursor.close()

            except Exception as e:
                dispatcher.utter_message(text=ERROR_MESSAGE)
                logging.error("Error during action code smell details: %s", e)
            finally:
                connection.close()  # Closing Connection
        else:
            dispatcher.utter_message(text=ERROR_MESSAGE)
            logging.error("Error during action code smell details: %s", CONNECTION_ERROR_MASSAGE)

        return []


class ActionExplainCodeSmellProblems(Action):

    def name(self) -> Text:
        return "action_explain_code_smell_problems"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        code_smell_id = next(tracker.get_latest_entity_values("code_smell_id"), None)
        code_smell_name = next(tracker.get_latest_entity_values("code_smell_name"), None)

        connection = get_connection()
        if connection:
            try:
                cursor = connection.cursor(dictionary=True)

                if code_smell_id:
                    query = "SELECT problems FROM codesmell WHERE id=%s"
                    cursor.execute(query, (code_smell_id,))
                    result = cursor.fetchone()
                elif code_smell_name:
                    query = "SELECT problems FROM codesmell WHERE name=%s"
                    best_match_name, similarity_score = code_smell_name_matcher(code_smell_name)
                    if similarity_score >= 70:
                        cursor.execute(query, (best_match_name,))
                    else:
                        cursor.execute(query, (code_smell_name,))
                    result = cursor.fetchone()
                else:
                    dispatcher.utter_message(
                        text="I'm sorry, I didn't understand what code smell you were referring to.")
                    return []

                if result:
                    code_smell_problems = result["problems"]
                    dispatcher.utter_message(text=f"{code_smell_problems}")
                else:
                    dispatcher.utter_message(text="Sorry, I couldn't find any details about that.")

                cursor.close()

            except Exception as e:
                dispatcher.utter_message(text=ERROR_MESSAGE)
                logging.error("Error during action code smell problems: %s", e)

            finally:
                connection.close()  # Closing Connection
        else:
            dispatcher.utter_message(text=ERROR_MESSAGE)
            logging.error("Error during action code smell problems: %s", CONNECTION_ERROR_MASSAGE)

        return []


class ActionExplainCodeSmellSolution(Action):

    def name(self) -> Text:
        return "action_explain_code_smell_solution"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        code_smell_id = next(tracker.get_latest_entity_values("code_smell_id"), None)
        code_smell_name = next(tracker.get_latest_entity_values("code_smell_name"), None)

        connection = get_connection()
        if connection:
            try:
                cursor = connection.cursor(dictionary=True)

                if code_smell_id:
                    query = "SELECT solution FROM codesmell WHERE id=%s"
                    cursor.execute(query, (code_smell_id,))
                    result = cursor.fetchone()
                elif code_smell_name:
                    query = "SELECT solution FROM codesmell WHERE name=%s"
                    best_match_name, similarity_score = code_smell_name_matcher(code_smell_name)
                    if similarity_score >= 70:
                        cursor.execute(query, (best_match_name,))
                    else:
                        cursor.execute(query, (code_smell_name,))
                    result = cursor.fetchone()
                else:
                    dispatcher.utter_message(
                        text="I'm sorry, I didn't understand what code smell you were referring to.")
                    return []

                if result:
                    code_smell_solution = result["solution"]
                    dispatcher.utter_message(text=f"{code_smell_solution}")
                else:
                    dispatcher.utter_message(text="Sorry, I couldn't find any details about that.")

                cursor.close()

            except Exception as e:
                dispatcher.utter_message(text=ERROR_MESSAGE)
                logging.error("Error during action code smell solution: %s", e)

            finally:
                connection.close()  # Closing Connection
        else:
            dispatcher.utter_message(text=ERROR_MESSAGE)
            logging.error("Error during action code smell solution: %s", CONNECTION_ERROR_MASSAGE)

        return []


class ActionProjectAnalysis(Action):

    def name(self) -> Text:
        return "action_project_analysis"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        github_repo_url = next(tracker.get_latest_entity_values("project_repository"), None)

        if not github_repo_url:
            dispatcher.utter_message(text="You haven`t provided a github repository url")
            return []

        url = os.getenv("PROJECT_ANALYZER_BASE_URL") + "/analyze-repository"
        payload = {
            "repository_url": github_repo_url
        }

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()  # raise an exepction if error code 4xx || 5xx
            analysis_result = response.json()
            dispatcher.utter_message(text=f"ANALISI COMPLETATA: {len(analysis_result)} code smells rilevati")

            directory = "./reports/"
            file_path = os.path.join(directory, "analysis_result.json")
            with open(file_path, "w") as file:
                json.dump(analysis_result, file)

        except Exception as e:
            dispatcher.utter_message(text=ERROR_MESSAGE)
            logging.error("Error during action project analysis: %s", e)

        return []
