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

logging.basicConfig(level=logging.INFO)
ERROR_MESSAGE = "Sorry, there was a problem... Please try again."

load_dotenv()
llm = os.getenv("LLM", "OpenAI")
if llm == "OpenAI":
    from llm.open_ai import complete_text
elif llm == "CodeLlama":
    from llm.codeLlama import complete_text


class ActionSuggestFix(Action):

    def name(self) -> Text:
        return "action_suggest_fix"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        report_filename = tracker.get_slot("report_filename")
        if report_filename is None:
            dispatcher.utter_message(text="It looks like you haven't done an analysis yet...")
            return []

        issue_index = next(tracker.get_latest_entity_values("issue_index"), None)
        if issue_index is None:
            dispatcher.utter_message(text="It looks like you haven't provided the issue's index...")
            return []

        cs_filename, cs_function_name, cs_name = get_info_from_report(report_filename, issue_index)
        if cs_filename is None or cs_function_name is None or cs_name is None:
            dispatcher.utter_message(text="It looks like you haven't provided a correct index...")
            return []

        url = os.getenv("PROJECT_ANALYZER_BASE_URL") + "/get-function-body"
        payload = {
            "filename": cs_filename,
            "function_name": cs_function_name
        }

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()  # raise an exception if error code 4xx || 5xx
            function_body = response.json()['function_body']
            cs_name = code_smell_name_matcher(cs_name)[0]
            cs_info = get_code_smell_by_name(cs_name, ["description", "problems", "solution"])

            prompt = []
            if llm == "OpenAI":
                prompt = [
                    {"role": "system", "content": f"{cs_info['description']} {cs_info['problems']} {cs_info['solution']}"},
                    {"role": "system", "content": f"{function_body}"},
                    {"role": "system",
                    "content": "You will be provided with the explanation of a code smell and the body of a function. Your task is to suggest how to fix the code smell in the function provided"},
                    {"role": "system",
                    "content": "You must write only the code, clearly indicating the modifications you have made using comments inside the code."},
                    {"role": "user",
                    "content": f"Suggest me how to fix the code smell {cs_name} in the function {cs_function_name}"},
                ]
            elif llm == "CodeLlama":
                prompt = [
                    {"role": "system", "content": f"{cs_info['description']} {cs_info['problems']} {cs_info['solution']}"},
                    {"role": "system", "content": f"{function_body}"},
                    {"role": "system",
                    "content": "You will be provided with the explanation of a code smell and the body of a function. Your task is to suggest how to fix the code smell in the function provided writing only the correct code and briefly explaining the changes made"},
                    {"role": "system",
                    "content": "You must write only the code, clearly indicating the modifications you have made using comments inside the code."},
                    {"role": "user",
                    "content": f"Suggest me how to fix the code smell {cs_name} in the function {cs_function_name}"},
                ]
            else:
                dispatcher.utter_message(text="There was a problema with the LLM... Please try again!")
                return[]

            suggestion = complete_text(prompt)
            suggestion = self._preserve_empty_lines(suggestion)

            # OpenAI API Response
            dispatcher.utter_message(
                text=f"Here's how you could fix the code smell \"{cs_name}\" within the function \"{cs_function_name}\":\n{suggestion}")

            # CodeLlama Response
            # dispatcher.utter_message(text=f"{suggestion}")
        except Exception as e:
            dispatcher.utter_message(text="An error occurred while suggesting the fix.")
            logging.error("Error during action suggest fix: %s", e)

        return []

    def _preserve_empty_lines(self, text: str) -> str:
        lines = text.split('\n')
        processed_lines = [line if line.strip() else ' ' for line in lines]  # Sostituisce righe vuote con uno spazio
        return '\n'.join(processed_lines)
