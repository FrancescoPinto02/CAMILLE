import ast
import os
import logging

#Logger Config
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_relative_path(full_path):
    parts = full_path.split(os.path.sep)
    try:
        projects_index = parts.index('projects')
        # Join the parts after 'projects'
        return os.path.sep.join(parts[projects_index + 1:])
    except ValueError:
        return full_path  # Return the full path if 'projects' is not found


def get_function_body(filename, function_name):
    # Construct the base path for project files
    project_base_path = os.path.join(os.path.dirname(__file__), '../input/projects/')
    file_path = os.path.join(project_base_path, filename)

    try:
        # Open the file and build AST
        with open(file_path, 'r') as file:
            tree = ast.parse(file.read(), filename=filename)

        # Traverse the AST
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                start_line = node.lineno
                end_line = node.body[-1].lineno

                # Open the file and extract function body
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                function_body = ''.join(lines[start_line - 1:end_line])
                return function_body
    except Exception as e:
        logging.error(f"Error analyzing the file '{file_path}': {e}")
        return None
    return None
