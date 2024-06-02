from flask import Flask, jsonify, request
import pandas as pd
import os
from cs_detection_tool.controller.analyzer import analyze_github_repository
from cs_detection_tool.utils.utils import extract_relative_path

app = Flask(__name__)


@app.route('/analyze-repository', methods=['POST'])
def analyze_repository():
    data = request.get_json()
    repository_url = data.get('repository_url')

    if not repository_url:
        return jsonify({"error": "Missing 'repository_url' parameter."}), 400

    result_file = analyze_github_repository(repository_url)
    if not result_file:
        return jsonify({"error": "Failed to analyze repository or no results found."}), 500

    df = pd.read_csv(result_file)

    # Restituisce solo il path relativo
    df['filename'] = df['filename'].apply(extract_relative_path)
    results = df.to_dict(orient='records')

    return jsonify(results), 200


if __name__ == '__main__':
    app.run(debug=True)
