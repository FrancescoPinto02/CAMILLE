import re
import ast


def get_variable_def(line):
    pattern = r'(\w)+(\[.*\])+\s*=\s*(\w*)'
    if re.match(pattern, line):
        # get the variable name
        variable = line.split('=')[0].strip().split('[')[0].strip()
        return variable
    return None


def get_all_set_variables(lines):
    variables = []
    for line in lines:
        variable = get_variable_def(line)
        if variable:
            variables.append(variable)
    return set(variables)


def search_variable_definition(var, fun_node, limit_node):
    # search for the variable definition
    # if the variable is defined in the same function, return the definition
    # if the variable is defined in the same file, return the definition
    # if the variable is defined in another file, return None
    # limit the node
    last_node_definition = None
    for node in ast.walk(fun_node):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == var:
                        last_node_definition = node
        if equal_node(node, limit_node):
            return last_node_definition
    return last_node_definition


def equal_node(node1, node2):
    if hasattr(node1, 'lineno') and hasattr(node2, 'lineno'):
        if node1.lineno == node2.lineno and node1.col_offset == node2.col_offset:
            return True
    return False
