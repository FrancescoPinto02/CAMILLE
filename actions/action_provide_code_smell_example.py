from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from utils.query import get_code_smell_by_id, get_code_smell_by_name

class ActionProvideCodeSmellExample(Action):

    def name(self) -> Text:
        return "action_provide_code_smell_example"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        code_smell_id = next(tracker.get_latest_entity_values("code_smell_id"), None)
        code_smell_name = next(tracker.get_latest_entity_values("code_smell_name"), None)
        result = None

        if code_smell_id:
            result = get_code_smell_by_id(code_smell_id, ["name", "bad_example", "good_example"])
        elif code_smell_name:
            result = get_code_smell_by_name(code_smell_name, ["name", "bad_example", "good_example"])
        else:
            dispatcher.utter_message(text="I'm sorry, I didn't understand what code smell you were referring to.")
            return []

        if result:
            bad_example = result['bad_example']
            good_example = result['good_example']

            # Rimuovi le righe vuote o sostituiscile con uno spazio
            bad_example = self._preserve_empty_lines(bad_example)
            good_example = self._preserve_empty_lines(good_example)

            dispatcher.utter_message(text=f"This is a code example with {result['name']}:\n{bad_example}")
            dispatcher.utter_message(text=f"And this is the correct one:\n{good_example}")
        else:
            dispatcher.utter_message(text="Sorry, I couldn't find any example about this code smell.")

        return []

    def _preserve_empty_lines(self, text: str) -> str:
        lines = text.split('\n')
        processed_lines = [line if line.strip() else ' ' for line in lines]  # Sostituisce righe vuote con uno spazio
        return '\n'.join(processed_lines)
