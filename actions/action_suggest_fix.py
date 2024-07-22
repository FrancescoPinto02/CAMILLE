from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import logging
from db.query import get_code_smell_by_name
import requests
from dotenv import load_dotenv
import os
from utils.report_utilities import get_info_from_report
from utils.string_matcher import code_smell_name_matcher
from llm.open_ai import complete_text

logging.basicConfig(level=logging.INFO)
ERROR_MESSAGE = "Sorry, there was a problem... Please try again."
load_dotenv()

class ActionSuggestFix(Action):

    def name(self) -> Text:
        return "action_suggest_fix"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Retrieve the report filename from the slot
        report_filename = tracker.get_slot("report_filename")
        if not report_filename:
            dispatcher.utter_message(text="It looks like you haven't done an analysis yet...")
            return []

        # Retrieve the issue index from the user's input
        issue_index = next(tracker.get_latest_entity_values("issue_index"), None)
        if not issue_index:
            dispatcher.utter_message(text="It looks like you haven't provided the issue's index...")
            return []

        # Extract code smell information from the report
        cs_filename, cs_function_name, cs_name = get_info_from_report(report_filename, issue_index)
        if not cs_filename or not cs_function_name or not cs_name:
            dispatcher.utter_message(text="It looks like you haven't provided a correct index...")
            return []

        # Prepare the request to get the function body
        url = os.getenv("PROJECT_ANALYZER_BASE_URL") + "/get-function-body"
        payload = {"filename": cs_filename, "function_name": cs_function_name}

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()  # Raise an exception for HTTP errors
            function_body = response.json().get('function_body')
            if not function_body:
                dispatcher.utter_message(text="Could not retrieve the function body.")
                return []

            # Get Code Smell info
            cs_name = code_smell_name_matcher(cs_name)[0]
            cs_info = get_code_smell_by_name(cs_name, ["description", "problems", "solution", "prompt_example"])

            # Prepare the prompt for the LLM
            prompt = self.create_prompt(cs_name, cs_info, cs_function_name, function_body)
            suggestion = complete_text(prompt)
            suggestion = self._preserve_empty_lines(suggestion)

            # Send the suggestion to the user
            dispatcher.utter_message(text=f"Here's how you could fix the code smell \"{cs_name}\" "
                                          f"within the function \"{cs_function_name}\":\n{suggestion}")

        except requests.RequestException as e:
            dispatcher.utter_message(text="An error occurred while retrieving the function body.")
            logging.error("Request error during action suggest fix: %s", e)
        except Exception as e:
            dispatcher.utter_message(text="An error occurred while suggesting the fix.")
            logging.error("Error during action suggest fix: %s", e)

        return []

    def create_prompt(self, cs_name: str, cs_info: Dict[Text, Any], cs_function_name: str,
                      function_body: str) -> List[Dict[str, str]]:
        """Create the prompt for the LLM based on the code smell information and function body."""
        return [
            {"role": "system", "content": f"I will provide you the description of the code smell called {cs_name}."},
            {"role": "system", "content": f"Context: {cs_info['description']} Problems: {cs_info['problems']} "
                                          f"Solution: {cs_info['solution']}"},

            {"role": "system", "content": f"Now I will provide you with an example of {cs_name}. In the example "
                                          f"the lines starting with '-' indicate smelly code, while the lines "
                                          f"starting with '+' indicate correct code."},
            {"role": "system", "content": f"Example: {cs_info['prompt_example']}"},

            {"role": "system", "content": f"Now I will provide you a function affected by {cs_name}."},
            {"role": "system", "content": f"{function_body}"},

            {"role": "system", "content": f"Your task is to refactor the function {cs_function_name} "
                                          f"and remove the code smell {cs_name}."},
            {"role": "system", "content": "You must write only the code of the whole refactored function clearly "
                                          "indicating the modifications you have done using comments inside the code."},
        ]

    def _preserve_empty_lines(self, text: str) -> str:
        """Preserve empty lines in the text by replacing them with a space."""
        lines = text.split('\n')
        processed_lines = [line if line.strip() else ' ' for line in lines]
        return '\n'.join(processed_lines)
