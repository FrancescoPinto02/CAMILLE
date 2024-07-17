from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import os
import requests
import csv
import logging


class ActionProjectAnalysis(Action):

    def name(self) -> Text:
        return "action_project_analysis"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Get slot value
        github_repo_url = tracker.get_slot("repo_url")

        if not github_repo_url:
            dispatcher.utter_message(text="You haven’t provided a GitHub repository URL.")
            return []

        # Request Preparation
        url = os.getenv("PROJECT_ANALYZER_BASE_URL") + "/analyze-repository"
        payload = {"repository_url": github_repo_url}

        try:
            # Send the request to the project analyzer service
            response = requests.post(url, json=payload)
            response.raise_for_status()  # Raise an exception for HTTP errors
            analysis_result = response.json()
            dispatcher.utter_message(text=f"Analysis Completed: {len(analysis_result)} code smells found!")

            # Save the analysis report
            report_filename = self._save_analysis_report(tracker.sender_id, analysis_result)
            dispatcher.utter_message(text="The report is ready for download.")
            return [SlotSet("report_filename", report_filename)]
        except requests.RequestException as e:
            logging.error("Request failed: %s", e)
            dispatcher.utter_message(
                text="Sorry, there was an error while analyzing the repository. Please try again later.")
        except Exception as e:
            logging.error("Error during project analysis: %s", e)
            dispatcher.utter_message(text="Sorry, something went wrong during the analysis.")

        return []

    def _save_analysis_report(self, user_id: str, analysis_result: List[Dict[Text, Any]]) -> str:
        # Ensure the reports directory exists
        directory = "./reports/"
        os.makedirs(directory, exist_ok=True)

        # Define the report file path
        report_filename = f"analysis_result_{user_id}.csv"
        file_path = os.path.join(directory, report_filename)

        # Write the analysis result to the CSV file
        with open(file_path, "w", newline='') as file:
            writer = csv.writer(file)
            if analysis_result:
                header = ['index'] + list(analysis_result[0].keys())
                writer.writerow(header)

            for i, item in enumerate(analysis_result, 1):
                if 'filename' in item:
                    # Normalize file paths
                    item['filename'] = item['filename'].replace("\\", "/")
                writer.writerow([i] + list(item.values()))

        return report_filename
