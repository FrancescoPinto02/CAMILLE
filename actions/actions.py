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
                    dispatcher.utter_message(text="Here is the list of code smells I can detect:")
                    for id, name in result:
                        dispatcher.utter_message(text=f"{id}: {name}")
                else:
                    dispatcher.utter_message(text="No code smells found in the database.")

                cursor.close()

            except mysql.connector.Error as err:
                dispatcher.utter_message(text=f"Database error: {err}")
            finally:
                connection.close()  # Closing Connection
        else:
            dispatcher.utter_message(text="Failed to connect to the database.")

        return []


class ActionProvideCodeSmellDetails(Action):

    def name(self) -> Text:
        return "action_provide_code_smell_details"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        code_smell_id = next(tracker.get_latest_entity_values("code_smell_id"), None)
        code_smell_name = next(tracker.get_latest_entity_values("code_smell_name"), None)

        if not code_smell_id and not code_smell_name:
            dispatcher.utter_message(text="I'm sorry, I didn't understand what code smell you were referring to.")
            return []

        connection = get_connection()
        if connection:
            try:
                cursor = connection.cursor()

                if code_smell_id:
                    query = "SELECT description FROM codesmell WHERE id=%s"
                    cursor.execute(query, (code_smell_id,))
                    result = cursor.fetchone()
                else:
                    query = "SELECT description FROM codesmell WHERE name=%s"
                    cursor.execute(query, (code_smell_name,))
                    result = cursor.fetchone()

                if result:
                    code_smell_details = result[0]
                    dispatcher.utter_message(text=f"{code_smell_details}")
                else:
                    dispatcher.utter_message(text="Sorry, I couldn't find any details about that.")

                cursor.close()

            except mysql.connector.Error as err:
                dispatcher.utter_message(text=f"Database error: {err}")
            finally:
                connection.close()  # Closing Connection
        else:
            dispatcher.utter_message(text="Failed to connect to the database.")

        return []
