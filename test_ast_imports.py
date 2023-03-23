import ast

def get_imports(file_path):
    with open(file_path) as f:
        node = ast.parse(f.read())
    
    imports = []
    for n in node.body:
        if isinstance(n, ast.Import):
            imports += [alias.name for alias in n.names]
        elif isinstance(n, ast.ImportFrom):
            imports += [n.module + '.' + alias.name for alias in n.names]
    return imports

print(get_imports('./parse_test_result.py'))
