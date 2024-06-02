import os


def extract_relative_path(full_path):
    parts = full_path.split(os.path.sep)
    try:
        projects_index = parts.index('projects')
        # Join the parts after 'projects'
        return os.path.sep.join(parts[projects_index + 1:])
    except ValueError:
        return full_path  # Return the full path if 'projects' is not found
