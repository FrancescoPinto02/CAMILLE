import logging
import os
import re

import pandas as pd
from urllib.parse import urlparse

from cs_detection_tool.controller.analyzer import analyze_project, analyze_github_repository, get_repo_details
from cs_detection_tool.utils.utils import get_function_body
from db.query import get_code_smell_by_name
from utils.function_replacer import replace_function_in_file
from utils.string_matcher import code_smell_name_matcher
from llm.open_ai import complete_text

logging.basicConfig(level=logging.INFO)


def run_test(max_projects_to_analyze=10):
    analyzed_projects = 0
    repo_testing_dataset = "cs_detection_tool/input/tests/NICHE.csv"
    df = pd.read_csv(repo_testing_dataset)

    for index, row in df.iterrows():
        repo_url = f"https://github.com/{row['GitHub_Repo']}.git"
        if not row['tested']:
            try:
                test_refactoring_single_project(repo_url)
                df.at[index, 'tested'] = True
                analyzed_projects += 1
                logging.info("Testing completed: %s", repo_url)
                logging.info("%s/%s", analyzed_projects, max_projects_to_analyze)
            except Exception as e:
                logging.error("Error refactoring testing of repo %s: %s", repo_url, e)

        if analyzed_projects >= max_projects_to_analyze:
            break

    df.to_csv(repo_testing_dataset, index=False)  # Salva il DataFrame aggiornato
    print("Test completati e stato salvato nel file CSV.")


def test_refactoring_single_project(repo_url):
    creator_name, repo_name = get_repo_details(repo_url)

    # Analizza il repository prima del refactoring
    input_path = "cs_detection_tool/input/tests/"
    output_path = "cs_detection_tool/output/tests/"
    before_refactoring_file = "beforeRefactoring.csv"

    res = analyze_github_repository(repo_url, input_path, output_path, before_refactoring_file)
    res = res.replace("\\", "/")

    res_df = pd.read_csv(res)

    for index, row in res_df.iterrows():
        filename = row['filename']
        function_name = row['function_name']
        cs_name = row['name_smell']

        # Ottieni informazioni dettagliate sul code smell
        cs_name = code_smell_name_matcher(cs_name)[0]
        cs_info = get_code_smell_by_name(cs_name, ["description", "problems", "solution", "prompt_example"])

        # Ottieni il corpo della funzione
        function_body = get_function_body(filename, function_name)

        # Crea il prompt per il refactoring
        prompt = create_prompt(cs_name, cs_info, function_body, function_name)

        # Ottieni il suggerimento per il refactoring
        suggestion = complete_text(prompt)
        # Pulisci e applica il refactoring alla funzione
        refactored_function = clean_python_script(suggestion)
        replace_function_in_file(filename, function_name, refactored_function)

    # Analizza il progetto dopo il refactoring
    analyze_project(f"{input_path}{creator_name}/{repo_name}/", f"{output_path}{creator_name}/{repo_name}/",
                    "afterRefactoring.csv")


def create_prompt(cs_name, cs_info, function_body, function_name):
    prompt = [
        {"role": "system", "content": f"I will provide you the description of the code smell called {cs_name}."},
        {"role": "system",
         "content": f"Context: {cs_info['description']} Problems: {cs_info['problems']} Solution: {cs_info['solution']}"},
        {"role": "system",
         "content": f"Now I will provide you with an example of {cs_name}. In the example, the lines starting with '-' indicate smelly code, whereas the lines starting with '+' indicate correct code."},
        {"role": "system", "content": f"Example: {cs_info['prompt_example']}"},
        {"role": "system", "content": f"Now I will provide you a function affected by {cs_name}."},
        {"role": "system", "content": f"{function_body}"},
        {"role": "system",
         "content": f"Your task is to refactor the function {function_name} and remove the code smell {cs_name}."},
        {"role": "system",
         "content": "You must write only the code of the whole refactored function without any comments"},
    ]
    return prompt


def clean_python_script(script):
    # Rimuovi backticks tripli che delimitano blocchi di codice
    script = re.sub(r'```python', '', script)
    script = re.sub(r'```', '', script)

    # Rimuovi eventuali spazi bianchi iniziali e finali
    script = script.strip()

    return script


if __name__ == "__main__":
    run_test(100)
