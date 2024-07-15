import ast
import os
import re

import astor
import pandas as pd
from urllib.parse import urlparse

from cs_detection_tool.controller.analyzer import analyze_project, analyze_github_repository
from cs_detection_tool.utils.utils import get_function_body
from db.query import get_code_smell_by_name
from utils.function_replacer import replace_function_in_file
from utils.string_matcher import code_smell_name_matcher
from llm.open_ai import complete_text

new_func_ex = '''
def example_function(new_arg1, new_arg2):
    print("This is the new function")
    return new_arg1 + new_arg2
'''

def run_test():
    test_repos_csv = f"cs_detection_tool/input/tests/github_repos.csv"
    test_input_path = f"cs_detection_tool/input/tests/"
    test_output_path = f"cs_detection_tool/output/tests"
    result_files = []

    repos_df = pd.read_csv(test_repos_csv)
    for index, row in repos_df.iterrows():
        repo_url = row['repo_url']
        res = analyze_github_repository(repo_url, test_input_path, test_output_path, "before_refactoring_results.csv")
        result_files.append(res.replace("\\", "/"))

    for res in result_files:
        res_df = pd.read_csv(res)
        for index, row in res_df.iterrows():
            filename = row['filename']
            function_name = row['function_name']
            cs_name = row['name_smell']
            cs_name = code_smell_name_matcher(cs_name)[0]
            cs_info = get_code_smell_by_name(cs_name, ["description", "problems", "solution"])
            function_body = get_function_body(filename, function_name)

            prompt = [
                {"role": "system", "content": f"{cs_info['description']} {cs_info['problems']} {cs_info['solution']}"},
                {"role": "system", "content": f"{function_body}"},
                {"role": "system",
                 "content": "You will be provided with the explanation of a code smell and the body of a function. Your task remove the code smell in the function provided."},
                {"role": "system",
                 "content": "Write only the correct function without any comments"},
                {"role": "user",
                 "content": f"Suggest me how to fix the code smell {cs_name} in the function {function_name}"},
            ]
            suggestion = complete_text(prompt)
            refactored_function = clean_python_script(suggestion)
            replace_function_in_file(filename, function_name, refactored_function)
            print(f"refactored {function_name} in {filename}")

    for index, row in repos_df.iterrows():
        repo_url = row['repo_url']
        parsed_url = urlparse(repo_url)
        path_parts = parsed_url.path.strip('/').split('/')
        creator_name = path_parts[0]
        repo_name = os.path.splitext(path_parts[1])[0]

        input_path = os.path.join(test_input_path, creator_name, repo_name)
        output_path = os.path.join(test_output_path, creator_name, repo_name)
        analyze_project(input_path, output_path, "after_refactoring_results.csv")

    return

"""
def replace_function_in_file(file_path, function_name, new_function_definition, encoding='utf-8'):
    # Read the source code
    with open(file_path, 'r', encoding=encoding) as file:
        source = file.read()

    # Parse the source code into an AST
    tree = ast.parse(source)

    # This class will be used to visit and replace the specified function
    class FunctionReplacer(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            # Check if this is the function to replace
            if node.name == function_name:
                # Parse new function definition from string to AST node
                new_function_tree = ast.parse(new_function_definition)
                if isinstance(new_function_tree, ast.Module) and len(new_function_tree.body) == 1 and isinstance(
                        new_function_tree.body[0], ast.FunctionDef):
                    return new_function_tree.body[0]  # Replace the entire function node
            return self.generic_visit(node)  # Continue traversing to find the function

    # Create an instance of our AST transformer and apply it
    replacer = FunctionReplacer()
    new_tree = replacer.visit(tree)

    # Write the modified AST back to the file
    with open(file_path, 'w', encoding=encoding) as file:
        file.write(astor.to_source(new_tree))
"""

def clean_python_script(script):
    # Rimuovere gli apici di blocco utilizzati per definire blocchi di codice in Markdown o simili
    script = re.sub(r"```python|```", "", script).strip()

    # Rimuovere tutte le linee che iniziano con 'import' o 'from ... import ...', mantenendo la formattazione
    cleaned_lines = []
    for line in script.splitlines():
        if line.strip().startswith(('import ', 'from ')):
            continue  # Salta le righe di import
        cleaned_lines.append(line)

    # Unire le righe pulite e gestire correttamente le linee vuote
    cleaned_script = '\n'.join(cleaned_lines)
    cleaned_script = re.sub(r'\n{3,}', '\n\n', cleaned_script)  # Riduce le linee vuote multiple a doppie linee vuote

    return cleaned_script


if __name__ == '__main__':
    res = analyze_github_repository("https://github.com/amaiya/ktrain.git", "cs_detection_tool/input/tests/", "cs_detection_tool/output/tests/", "beforeRefactoring.csv")
    res = res.replace("\\", "/")

    res_df = pd.read_csv(res)
    for index, row in res_df.iterrows():
        filename = row['filename']
        function_name = row['function_name']
        cs_name = row['name_smell']
        cs_name = code_smell_name_matcher(cs_name)[0]
        cs_info = get_code_smell_by_name(cs_name, ["description", "problems", "solution", "prompt_example"])
        function_body = get_function_body(filename, function_name)
        prompt = [
            {"role": "system", "content": f"I will provide you the description of the code smell called {cs_name}."},
            {"role": "system", "content": f"Context:{cs_info['description']} Problems:{cs_info['problems']} Solution:{cs_info['solution']}"},
            {"role": "system", "content": f"Now I will provide you with an example of {cs_name}. In the example the lines starting with '-' indicates smelly code, instead the lines starting with '+' indicates correct code."},
            {"role": "system", "content": f"Example:{cs_info['prompt_example']}"},
            {"role": "system", "content": f"Now I will provide you a function affected by {cs_name}."},
            {"role": "system", "content": f"{function_body}"},
            {"role": "system",
             "content": f"Your task is to refactor the function {function_name} and remove the code smell {cs_name}."},
            {"role": "system",
             "content": "you must write the whole refactored function without any comments"},
        ]
        print(prompt)
        suggestion = complete_text(prompt)
        print(suggestion)
        refactored_function = clean_python_script(suggestion)
        print(refactored_function)
        replace_function_in_file(filename, function_name, refactored_function)
        print(f"refactored {function_name} in {filename}")

    analyze_project("cs_detection_tool/input/tests/amaiya/ktrain/", "cs_detection_tool/output/tests/amaiya/ktrain/", "afterRefactoring.csv")
