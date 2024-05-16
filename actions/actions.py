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
                connection.close() # Closing Connection
        else:
            dispatcher.utter_message(text="Failed to connect to the database.")

        return []
