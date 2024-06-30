import os
from urllib.parse import urlparse
import pandas as pd
from git import Repo
from cs_detection_tool.components import detector


def get_python_files(path):
    result = []
    if os.path.isfile(path):
        if path.endswith(".py"):
            result.append(path)
            return result
    for root, dirs, files in os.walk(path):
        if "venv" in dirs:
            dirs.remove("venv")
        if "lib" in dirs:
            dirs.remove("lib")
        for file in files:
            if file.endswith(".py"):
                result.append(os.path.abspath(os.path.join(root, file)))
    return result


def analyze_project(project_path, output_path="."):
    col = ["filename", "function_name", "smell", "name_smell", "message"]
    to_save = pd.DataFrame(columns=col)
    filenames = get_python_files(project_path)

    for filename in filenames:
        if "tests/" not in filename:  # ignore test files
            try:
                result = detector.inspect(filename, output_path)
                to_save = to_save.merge(result, how='outer')
            except SyntaxError as e:
                message = e.msg
                error_path = output_path
                if not os.path.exists(error_path):
                    os.makedirs(error_path)
                with open(f"{error_path}/error.txt", "a") as error_file:
                    error_file.write(message)
                continue
            except FileNotFoundError as e:
                message = e
                error_path = output_path
                if not os.path.exists(error_path):
                    os.makedirs(error_path)
                with open(f"{error_path}/error.txt", "a") as error_file:
                    error_file.write(str(message))
                continue

    to_save.to_csv(output_path + "/to_save.csv", index=False, mode='w')


def analyze_github_repository(repo_url):
    # Ottieni il nome del creatore e del repository dall'URL
    parsed_url = urlparse(repo_url)
    path_parts = parsed_url.path.strip('/').split('/')
    creator_name = path_parts[0]
    repo_name = os.path.splitext(path_parts[1])[0]

    # Percorsi di input e output
    input_path = f"cs_detection_tool/input/projects/{creator_name}/{repo_name}"
    output_path = f"cs_detection_tool/output/projects_analysis/{creator_name}/{repo_name}"
    result_file = os.path.join(output_path, "to_save.csv")

    try:
        if os.path.exists(input_path) and os.path.exists(output_path):
            # Se la repository è già stata clonata e l'output è presente, esegui un pull
            repo = Repo(input_path)
            origin = repo.remotes.origin
            origin.pull()
            print(f"Pulling {repo_name}")
        else:
            # Altrimenti, clona la repository
            Repo.clone_from(repo_url, input_path)
            print(f"Cloned {repo_name}")

            # Crea la cartella per l'analisi dei progetti
            os.makedirs(output_path, exist_ok=True)

        # Avvia l'analisi del progetto
        analyze_project(input_path, output_path)
        return result_file if os.path.exists(result_file) else None

    except Exception as e:
        print(f"Error analyzing repository '{repo_name}': {e}")
        return None
