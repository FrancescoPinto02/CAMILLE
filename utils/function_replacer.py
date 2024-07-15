import ast
import astor


class FunctionReplacer(ast.NodeTransformer):
    def __init__(self, function_name, new_function):
        self.function_name = function_name
        # Parse della nuova funzione come un albero AST
        self.new_function = ast.parse(new_function).body[0]

    def visit_FunctionDef(self, node):
        # Sostituisce la funzione solo se il nome corrisponde
        if node.name == self.function_name:
            return self.new_function
        return node


def replace_function_in_file(file_name, function_name, new_function):
    # Legge il contenuto del file e lo converte in un AST
    with open(file_name, 'r', encoding='utf-8') as file:
        tree = ast.parse(file.read(), filename=file_name)

    # Crea e applica il transformer
    replacer = FunctionReplacer(function_name, new_function)
    new_tree = replacer.visit(tree)

    # Scrive il nuovo albero AST nel file con encoding UTF-8
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(astor.to_source(new_tree))

