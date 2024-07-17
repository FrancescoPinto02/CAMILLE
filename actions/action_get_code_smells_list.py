from typing import Any, Text, Dict, List, Optional
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from db.query import get_all_code_smells
import logging


class ActionGetCodeSmellsList(Action):

    def name(self) -> Text:
        return "action_get_code_smells_list"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        try:
            # Retrieve all the code smells
            results = get_all_code_smells(["id", "name"])
        except Exception as e:
            logging.error(f"Failed to fetch code smells: {e}")
            dispatcher.utter_message(text="Sorry, something went wrong while fetching the code smells...")
            return []

        if results:
            # Build the response
            message_lines = ["Here is the list of code smells:"]
            for code_smell in results:
                message_lines.append(f"{code_smell['id']}: {code_smell['name']}")
            message_lines.append("\nYou can ask me for more information or examples about each code smell in the list.")
            message = "\n".join(message_lines)
            dispatcher.utter_message(text=message)
        else:
            dispatcher.utter_message(text="No code smells found...")

        return []
