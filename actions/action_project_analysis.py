import csv
import os
import logging
import requests
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
ERROR_MESSAGE = "Sorry, there was a problem... Please try again."


class ActionProjectAnalysis(Action):

    def name(self) -> Text:
        return "action_project_analysis"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        github_repo_url = next(tracker.get_latest_entity_values("project_repository"), None)

        if not github_repo_url:
            dispatcher.utter_message(text="You haven’t provided a github repository url")
            return []

        url = os.getenv("PROJECT_ANALYZER_BASE_URL") + "/analyze-repository"
        payload = {
            "repository_url": github_repo_url
        }

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()  # raise an exception if error code 4xx || 5xx
            analysis_result = response.json()
            dispatcher.utter_message(text=f"Analysis Completed: {len(analysis_result)} code smells found!")

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

                dispatcher.utter_message(text=f"The report is ready for the download.")
                return [SlotSet("report_filename", report_filename)]
        except Exception as e:
            dispatcher.utter_message(text=ERROR_MESSAGE)
            logging.error("Error during action project analysis: %s", e)

        return []
