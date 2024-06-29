from typing import Any, Text, Dict, List
import pandas as pd
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import logging
from jinja2 import Template
from dotenv import load_dotenv
import os

logging.basicConfig(level=logging.INFO)
ERROR_MESSAGE = "Sorry, there was a problem... Please try again."
load_dotenv()


class ActionAnalysisReport(Action):

    def name(self) -> Text:
        return "action_analysis_report"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        report_filename = tracker.get_slot("report_filename")

        if report_filename is None:
            dispatcher.utter_message(text="It looks like you haven't done an analysis yet...")
            return []

        directory = "./reports/"
        file_path = os.path.join(directory, report_filename)

        # Read the report file
        report_df = pd.read_csv(file_path)

        # Generate the HTML report
        html_report = self.generate_html_report(report_df)

        dispatcher.utter_message(text=html_report)
        dispatcher.utter_message(text="Feel free to ask for suggestions about issues fixing. Remember to provide me issue`s index!")
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
                    white-space: nowrap; /* Impedisce il testo di andare a capo */
                    overflow: hidden; /* Nasconde il testo che supera la larghezza */
                    text-overflow: ellipsis; /* Mostra i puntini di sospensione (...) se necessario */
                }
                .table thead th:nth-child(1),
                .table tbody td:nth-child(1) {
                    width: 5%; /* Larghezza della prima colonna (Issue) */
                }
                .table thead th:nth-child(2),
                .table tbody td:nth-child(2) {
                    width: 20%; /* Larghezza della seconda colonna (Filename) */
                }
                .table thead th:nth-child(3),
                .table tbody td:nth-child(3) {
                    width: 20%; /* Larghezza della terza colonna (Function Name) */
                }
                .table thead th:nth-child(4),
                .table tbody td:nth-child(4) {
                    width: 25%; /* Larghezza della quarta colonna (Message) */
                }
                .table thead th:nth-child(5),
                .table tbody td:nth-child(5) {
                    width: 30%; /* Larghezza della quinta colonna (Smell Name) */
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
