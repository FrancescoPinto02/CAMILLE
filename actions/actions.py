# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions
import csv
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher
import logging

from llm.open_ai import complete_text
from utils.query import get_all_code_smells
from utils.query import get_code_smell_by_id
from utils.query import get_code_smell_by_name
import requests
from dotenv import load_dotenv
import os

from utils.report_utilities import get_info_from_report
from utils.string_matcher import code_smell_name_matcher

logging.basicConfig(level=logging.INFO)

ERROR_MESSAGE = "Sorry, there was a problem... Please try again."

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

        results = get_all_code_smells(["id", "name"])

        if results:
            message = "Here is the list of code smells:\n"
            for code_smell in results:
                message += f"{code_smell['id']}: {code_smell['name']}\n"
            dispatcher.utter_message(text=message)
        else:
            dispatcher.utter_message(text="No code smells found in the database.")

        return []


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
            if not os.path.exists(directory):
                os.makedirs(directory)

            user_id = tracker.sender_id
            report_filename = f"analysis_result_{user_id}.csv"
            file_path = os.path.join(directory, report_filename)

            with open(file_path, "w", newline='') as file:
                writer = csv.writer(file)

                if analysis_result:
                    header = ['index'] + list(analysis_result[0].keys())
                    writer.writerow(header)

                for i, item in enumerate(analysis_result, 1):
                    # Normalize file paths to use forward slashes
                    if 'filename' in item:
                        item['filename'] = item['filename'].replace("\\", "/")
                    writer.writerow([i] + list(item.values()))

                return [SlotSet("report_filename", report_filename)]
        except Exception as e:
            dispatcher.utter_message(text=ERROR_MESSAGE)
            logging.error("Error during action project analysis: %s", e)

        return []


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

            prompt = [
                {"role": "system", "content": f"{cs_info['description']} {cs_info['problems']} {cs_info['solution']}"},
                {"role": "system", "content": f"{function_body}"},
                {"role": "system",
                 "content": "You will be provided with the explanation of a code smell and the body of a function. Your task is to suggest how to fix the code smell provided in the function provided"},
                {"role": "system",
                 "content": "You must write only the code, clearly indicating the modifications you have made using comments inside the code."},
                {"role": "user",
                 "content": f"Suggest me how to fix the code smell {cs_name} in the function {cs_function_name}"},
            ]
            suggestion = complete_text(prompt)
            suggestion = self._preserve_empty_lines(suggestion)

            dispatcher.utter_message(text=f"Here's how you could fix the code smell \"{cs_name}\" within the function \"{cs_function_name}\":\n{suggestion}")
        except Exception as e:
            dispatcher.utter_message(text="An error occurred while suggesting the fix.")
            logging.error("Error during action suggest fix: %s", e)

        return []

    def _preserve_empty_lines(self, text: str) -> str:
        lines = text.split('\n')
        processed_lines = [line if line.strip() else ' ' for line in lines]  # Sostituisce righe vuote con uno spazio
        return '\n'.join(processed_lines)
