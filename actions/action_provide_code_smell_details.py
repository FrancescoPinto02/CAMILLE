from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from utils.query import get_code_smell_by_id, get_code_smell_by_name


class ActionProvideCodeSmellDetails(Action):

    def name(self) -> Text:
        return "action_provide_code_smell_details"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        code_smell_id = next(tracker.get_latest_entity_values("code_smell_id"), None)
        code_smell_name = next(tracker.get_latest_entity_values("code_smell_name"), None)
        result = None

        if code_smell_id:
            result = get_code_smell_by_id(code_smell_id, ["description", "problems", "solution"])
        elif code_smell_name:
            result = get_code_smell_by_name(code_smell_name, ["description", "problems", "solution"])
        else:
            dispatcher.utter_message(text="I'm sorry, I didn't understand what code smell you were referring to.")
            return []

        if result:
            dispatcher.utter_message(text=f"{result['description']}")
            dispatcher.utter_message(text=f"{result['problems']}")
            dispatcher.utter_message(text=f"{result['solution']}")
        else:
            dispatcher.utter_message(text="Sorry, I couldn't find any details about this code smell.")

        return []
