import csv
import os.path


def get_info_from_report(filename, index):
    index = int(index)
    filename = str(filename).strip()
    report_path = os.path.join("reports", filename)

    with open(report_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if int(row['index']) == index:
                return row['filename'], row['function_name'], row['name_smell']

    return None, None, None

