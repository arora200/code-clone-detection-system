import ast
import random
import string

class CodeMutator(ast.NodeTransformer):
    """
    An AST visitor to perform various code mutations for generating clones.
    """
    def __init__(self):
        self.name_map = {}
        self.functions_visited = []

    def _get_new_name(self, old_name):
        if old_name not in self.name_map:
            new_name = ''.join(random.choices(string.ascii_lowercase, k=len(old_name)))
            # Ensure new name is not the same as old, and not already used
            while new_name == old_name or new_name in self.name_map.values():
                new_name = ''.join(random.choices(string.ascii_lowercase, k=len(old_name)))
            self.name_map[old_name] = new_name
        return self.name_map[old_name]

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Load)):
            if node.id not in ['True', 'False', 'None']: # Avoid renaming built-ins
                node.id = self._get_new_name(node.id)
        return node

    def visit_FunctionDef(self, node):
        # Rename function name
        node.name = self._get_new_name(node.name)
        # Rename arguments
        if node.args.args:
            for arg in node.args.args:
                arg.arg = self._get_new_name(arg.arg)
        self.generic_visit(node)
        self.functions_visited.append(node)
        return node

def rename_variables(code):
    """
    Renames variables, functions, and arguments in a piece of code.
    """
    try:
        tree = ast.parse(code)
        mutator = CodeMutator()
        mutated_tree = mutator.visit(tree)
        return ast.unparse(mutated_tree)
    except Exception:
        return code # Return original code if parsing fails

def add_redundant_statement(code):
    """
    Adds a redundant statement to a function in the code.
    """
    try:
        tree = ast.parse(code)
        mutator = CodeMutator()
        mutator.visit(tree) # Visit to collect function nodes
        
        if mutator.functions_visited:
            # Select a random function to modify
            func_to_modify = random.choice(mutator.functions_visited)
            
            # Create a redundant statement (e.g., a pass statement)
            redundant_stmt = ast.Pass()
            
            # Insert the statement at the beginning of the function body
            func_to_modify.body.insert(0, redundant_stmt)
            
            return ast.unparse(tree)
    except Exception:
        pass
    return code

def mutate_code(code):
    """
    Applies a random mutation to the given code.
    """
    mutations = [rename_variables, add_redundant_statement]
    mutation_to_apply = random.choice(mutations)
    return mutation_to_apply(code)

if __name__ == '__main__':
    sample_code = """
def my_function(a, b):
    result = a + b
    return result

x = 10
y = 20
z = my_function(x, y)
print(z)
"""
    print("--- Original Code ---")
    print(sample_code)
    
    print("\n--- Mutated Code (Renamed) ---")
    mutated = rename_variables(sample_code)
    print(mutated)
    
    print("\n--- Mutated Code (Redundant Stmt) ---")
    mutated = add_redundant_statement(sample_code)
    print(mutated)
