from typing import Any, Text, Dict, List, Optional
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from db.query import get_code_smell_by_id, get_code_smell_by_name
import logging


class ActionProvideCodeSmellDetails(Action):

    def name(self) -> Text:
        return "action_provide_code_smell_details"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Get entities values
        code_smell_id = next(tracker.get_latest_entity_values("code_smell_id"), None)
        code_smell_name = next(tracker.get_latest_entity_values("code_smell_name"), None)
        result = None

        try:
            if code_smell_id:
                # Query by ID
                result = get_code_smell_by_id(code_smell_id, ["description", "problems", "solution"])
            elif code_smell_name:
                # Query by Name
                result = get_code_smell_by_name(code_smell_name, ["description", "problems", "solution"])
            else:
                dispatcher.utter_message(text="I'm sorry, I didn't understand what code smell you are referring to...")
                return []
        except Exception as e:
            logging.error(f"Failed to fetch code smell details: {e}")
            dispatcher.utter_message(text="Sorry, something went wrong while fetching the code smell details...")
            return []

        if result:
            message = (
                f"{result['description']}\n\n"
                f"{result['problems']}\n\n"
                f"{result['solution']}"
            )
            dispatcher.utter_message(text=message)
        else:
            dispatcher.utter_message(text="Sorry, I couldn't find any details about this code smell.")

        return []
