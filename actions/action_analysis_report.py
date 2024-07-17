from typing import Any, Text, Dict, List
import pandas as pd
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import logging
from jinja2 import Template
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
ERROR_MESSAGE = "Sorry, there was a problem... Please try again."
load_dotenv()


class ActionAnalysisReport(Action):

    def name(self) -> Text:
        return "action_analysis_report"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Retrieve the report filename from the slot
        report_filename = tracker.get_slot("report_filename")

        if not report_filename:
            dispatcher.utter_message(text="It looks like you haven't done an analysis yet...")
            return []

        directory = "./reports/"
        file_path = os.path.join(directory, report_filename)

        try:
            # Read the report file
            report_df = pd.read_csv(file_path)

            # Generate the HTML report
            html_report = self.generate_html_report(report_df)

            dispatcher.utter_message(text=html_report)
            dispatcher.utter_message(
                text="Feel free to ask for suggestions about issues fixing. Remember to provide me the issue's index!")
        except FileNotFoundError:
            dispatcher.utter_message(text="Sorry, the report file could not be found.")
            logging.error(f"Report file {file_path} not found.")
        except pd.errors.EmptyDataError:
            dispatcher.utter_message(text="Sorry, the report file is empty.")
            logging.error(f"Report file {file_path} is empty.")
        except Exception as e:
            dispatcher.utter_message(text=ERROR_MESSAGE)
            logging.error(f"Error while processing the report: {e}")

        return []

    def generate_html_report(self, report_df: pd.DataFrame) -> Text:
        # Define the Jinja template
        template = Template("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Analysis Report</title>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
            <style>
                body { padding: 20px; font-size: 12px; }
                .table { margin-top: 20px; }
                .table th, .table td {
                    white-space: nowrap; /* Prevent text wrapping */
                    overflow: hidden; /* Hide overflow text */
                    text-overflow: ellipsis; /* Show ellipsis for overflow text */
                }
                .table thead th:nth-child(1),
                .table tbody td:nth-child(1) {
                    width: 5%; /* Width of the first column (Issue) */
                }
                .table thead th:nth-child(2),
                .table tbody td:nth-child(2) {
                    width: 20%; /* Width of the second column (Filename) */
                }
                .table thead th:nth-child(3),
                .table tbody td:nth-child(3) {
                    width: 20%; /* Width of the third column (Function Name) */
                }
                .table thead th:nth-child(4),
                .table tbody td:nth-child(4) {
                    width: 25%; /* Width of the fourth column (Message) */
                }
                .table thead th:nth-child(5),
                .table tbody td:nth-child(5) {
                    width: 30%; /* Width of the fifth column (Smell Name) */
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="my-4">Analysis Report</h1>
                <table class="table table-bordered">
                    <thead class="thead-dark">
                        <tr>
                            <th>Issue</th>
                            <th>Filename</th>
                            <th>Function Name</th>
                            <th>Message</th>
                            <th>Smell Name</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for row in report %}
                        <tr>
                            <td>{{ row.index }}</td>
                            <td>{{ row.filename }}</td>
                            <td>{{ row.function_name }}</td>
                            <td>{{ row.message }}</td>
                            <td>{{ row.name_smell }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.16.0/umd/popper.min.js"></script>
            <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
        </body>
        </html>
        """)

        # Render the template with data
        html_content = template.render(report=report_df.to_dict(orient='records'))

        return html_content
