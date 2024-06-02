import ast


def extract_libraries(tree):
    """
    Given a tree obtained from ast.parse() command, extract a list of libraries used in the tree.
    """
    libraries = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    libraries.append(alias.name + ' as ' + alias.asname)
                else:
                    libraries.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                if node.module != "*":
                    module_name = node.module
                    # if node.asname:
                    # module_name += ' as ' + node.asname
                    for alias in node.names:
                        if alias.asname:
                            libraries.append(module_name + '.' + alias.name + ' as ' + alias.asname)
                        else:
                            libraries.append(module_name + '.' + alias.name)
                else:
                    libraries.append(node.module)

    return set(libraries)


def extract_library_name(library):
    if "as" not in library:
        return library
    else:
        return library.split(" as ")[0]


def extract_library_as_name(library):
    if " as " not in library:
        return library
    else:
        return library.split(" as ")[1]


def get_library_of_node(node, libraries):
    """
    Given a node and a list of libraries, return the library of the node.
    """
    from_object = False
    n = node
    if isinstance(n, ast.Call):
        n = n.func
        while isinstance(n, ast.Attribute):
            from_object = True
            n = n.value
        if isinstance(n, ast.Name):
            method_name = n.id
        else:
            method_name = ""
        for lib in libraries:
            if method_name in lib:
                return lib
    if from_object:
        return "Unknown"
    else:
        return None
