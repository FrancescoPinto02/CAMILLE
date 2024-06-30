from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from db.query import get_all_code_smells

class ActionGetCodeSmellsList(Action):

    def name(self) -> Text:
        return "action_get_code_smells_list"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        results = get_all_code_smells(["id", "name"])

        if results:
            message = "Here is the list of code smells:\n"
            for code_smell in results:
                message += f"{code_smell['id']}: {code_smell['name']}\n"
            dispatcher.utter_message(text=message)
            dispatcher.utter_message(text="You can ask me for more information or examples about each code smell in the list.")
        else:
            dispatcher.utter_message(text="No code smells found.")

        return []