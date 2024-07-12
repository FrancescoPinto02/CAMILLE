import os
from urllib.parse import urlparse
import pandas as pd
from git import Repo, GitCommandError
from cs_detection_tool.components import detector
import logging

# Logger Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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


def analyze_project(project_path, output_path=".", result_filename="to_save.csv"):
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

    result_file_path = os.path.join(output_path, result_filename)
    to_save.to_csv(result_file_path, index=False, mode='w')


def get_repo_details(repo_url):
    parsed_url = urlparse(repo_url)
    path_parts = parsed_url.path.strip('/').split('/')
    if len(path_parts) < 2:
        raise ValueError("Invalid repository path in URL")
    creator_name = path_parts[0]
    repo_name = os.path.splitext(path_parts[1])[0]
    return creator_name, repo_name


def clone_or_pull_repo(repo_url, input_path):
    try:
        if os.path.exists(input_path):
            repo = Repo(input_path)
            origin = repo.remotes.origin
            origin.pull()
            logging.info(f"Pulled updates for repository {repo_url}")
        else:
            Repo.clone_from(repo_url, input_path)
            logging.info(f"Cloned repository {repo_url}")
    except GitCommandError as e:
        logging.error(f"Git command error: {e}")
        raise


def analyze_github_repository(repo_url, input_base='cs_detection_tool/input/projects/',
                              output_base='cs_detection_tool/output/projects_analysis/',
                              result_filename="to_save.csv"):
    try:
        # Obtain Repo Details
        creator_name, repo_name = get_repo_details(repo_url)

        # Build Input/Output Paths
        input_path = os.path.join(input_base, creator_name, repo_name)
        output_path = os.path.join(output_base, creator_name, repo_name)
        result_file = os.path.join(output_path, result_filename)

        # Clone or Pull the Repo
        clone_or_pull_repo(repo_url, input_path)

        # Create Project Analysis Folder
        os.makedirs(output_path, exist_ok=True)

        # Run the Project Analysis
        analyze_project(input_path, output_path, result_filename)

        if os.path.exists(result_file):
            return result_file
        else:
            logging.warning(f"Result file {result_filename} does not exist after analysis")
            return None

    except Exception as e:
        logging.error(f"Error analyzing repository '{repo_url}': {e}")
        return None
