# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions
import csv
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import logging
from utils.query import get_all_code_smells
from utils.query import get_code_smell_by_id
from utils.query import get_code_smell_by_name
import requests
from dotenv import load_dotenv
import os

logging.basicConfig(level=logging.INFO)

ERROR_MESSAGE = "Sorry, there was a problem... Please try again."

load_dotenv()


class ActionDefaultFallback(Action):

    def name(self) -> Text:
        return "action_default_fallback"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="utter_default")
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
            dispatcher.utter_message(text=f"This is a code example with {result['name']}:")
            dispatcher.utter_message(text=f"{result['bad_example']}")
            dispatcher.utter_message(text=f"And this is the corrected version:")
            dispatcher.utter_message(text=f"{result['good_example']}")
        else:
            dispatcher.utter_message(text="Sorry, I couldn't find any example about this code smell.")

        return []


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
            file_path = os.path.join(directory, f"analysis_result_{user_id}.csv")

            with open(file_path, "w", newline='') as file:
                writer = csv.writer(file)

                if analysis_result:
                    header = analysis_result[0].keys()
                    writer.writerow(header)
                for item in analysis_result:
                    writer.writerow(item.values())

        except Exception as e:
            dispatcher.utter_message(text=ERROR_MESSAGE)
            logging.error("Error during action project analysis: %s", e)

        return []
