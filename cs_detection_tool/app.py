from flask import Flask, jsonify, request
import pandas as pd
import os
from cs_detection_tool.controller.analyzer import analyze_github_repository
from cs_detection_tool.utils.utils import extract_relative_path, get_function_body

app = Flask(__name__)


@app.route('/analyze-repository', methods=['POST'])
def analyze_repository():
    data = request.get_json()
    repository_url = data.get('repository_url')

    if not repository_url:
        return jsonify({"error": "Missing 'repository_url' parameter."}), 400

    # Run repository analysis
    result_file = analyze_github_repository(repository_url)
    if not result_file:
        return jsonify({"error": "Failed to analyze repository or no results found."}), 500

    df = pd.read_csv(result_file)

    # Return only Relative Paths
    df['filename'] = df['filename'].apply(extract_relative_path)
    results = df.to_dict(orient='records')

    return jsonify(results), 200


@app.route('/get-function-body', methods=['POST'])
def get_function_body_endpoint():
    data = request.get_json()
    filename = data.get('filename')
    function_name = data.get('function_name')

    if not filename or not function_name:
        return jsonify({"error": "Missing 'filename' or 'function_name' parameter"}), 400

    # Retrieve function body
    function_body = get_function_body(filename, function_name)

    if function_body:
        return jsonify({"function_body": function_body})
    else:
        return jsonify({"error": "Function not found"}), 404


if __name__ == '__main__':
    app.run(debug=True)
