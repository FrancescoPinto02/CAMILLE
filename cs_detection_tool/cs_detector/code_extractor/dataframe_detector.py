import ast

import pandas as pd


def search_pandas_library(libraries):
    for lib in libraries:
        if 'pandas' in lib:
            short = extract_lib_object(lib)
            if short is None:
                short = 'pandas'
            return short
    return None


def load_dataframe_dict(path):
    return pd.read_csv(path, dtype={'id': 'string', 'library': 'string', 'method': 'string'})


def dataframe_check(fun_node, libraries, df_dict):
    short = search_pandas_library(libraries)
    list = [short]
    if short is None:
        return None
    return recursive_search_variables(fun_node, list, df_dict)
    # extract_variables([short])


def recursive_search_variables(fun_node, init_list, df_dict):
    list = init_list.copy()
    for node in ast.walk(fun_node):
        if isinstance(node, ast.Assign):
            # check if the right side contains a dataframe
            if isinstance(node.value, ast.Expr):
                expr = node.value
                if isinstance(expr.value, ast.Name):
                    name = expr.value
                    if name.id in list:
                        if hasattr(node.targets[0], 'id'):
                            if node.targets[0].id not in list:
                                list.append(node.target.id)
            if isinstance(node.value, ast.Name):

                name = node.value
                if name.id in list:
                    if hasattr(node.targets[0], 'id'):
                        if node.targets[0].id not in list:
                            list.append(node.targets[0].id)

            if isinstance(node.value, ast.Call):
                name_func = node.value.func
                if isinstance(name_func, ast.Attribute):
                    id = name_func.value
                    if isinstance(name_func.value, ast.Subscript):
                        if isinstance(name_func.value.value, ast.Name):
                            id = name_func.value.value.id
                    else:
                        if (isinstance(name_func.value, ast.Name)):
                            id = name_func.value.id
                        else:
                            continue
                    if id in list:
                        if name_func.attr in df_dict['method'].tolist():
                            if hasattr(node.targets[0], 'id'):
                                if node.targets[0].id not in list:
                                    list.append(node.targets[0].id)
            else:
                if isinstance(node.value, ast.Subscript):
                    if isinstance(node.value.value, ast.Name):
                        if node.value.value.id in list:
                            if hasattr(node.targets[0], 'id'):
                                if node.targets[0].id not in list:
                                    list.append(node.targets[0].id)
    if list == init_list:
        return list
    else:
        return recursive_search_variables(fun_node, list, df_dict)


def extract_lib_object(lib):
    try:
        split_lib = lib.split(" as ")
        if split_lib[1] is not None and split_lib[
            1] != "":  # this because some libraries are imported with and endwith as
            short = split_lib[1]
            return short
        else:
            return None
    except:
        return None


def extract_variables(list_variables):
    pass
