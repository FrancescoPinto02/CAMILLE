import ast
import os


def extract_relative_path(full_path):
    parts = full_path.split(os.path.sep)
    try:
        projects_index = parts.index('projects')
        # Join the parts after 'projects'
        return os.path.sep.join(parts[projects_index + 1:])
    except ValueError:
        return full_path  # Return the full path if 'projects' is not found


def get_function_body(filename, function_name):
    project_base_path = os.path.join(os.path.dirname(__file__), '../input/projects/')
    file_path = os.path.join(project_base_path, filename)

    try:
        with open(file_path, 'r') as file:
            tree = ast.parse(file.read(), filename=filename)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                start_line = node.lineno
                end_line = node.body[-1].lineno
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                function_body = ''.join(lines[start_line - 1:end_line])
                return function_body
    except Exception as e:
        print(f"Errore durante l'analisi del file: {e}")
        return None

    return None