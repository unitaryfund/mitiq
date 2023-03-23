import xmltodict
import ast
import os
import glob
import sys

with open('test-results.xml', 'r') as f:
    xml_data = f.read()

test_results = xmltodict.parse(xml_data)

# Sort the tests based on their execution time
sorted_tests = sorted(test_results['testsuites']['testsuite']['testcase'],
                      key=lambda x: float(x['@time']))

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

# Modify the source code files to add the `@pytest.mark.order` decorator
for i, test in enumerate(sorted_tests, start=1):
    test_name = test['@classname'] + '.' + test['@name']
    file_path = os.path.join(test['@classname'].replace('.', '/') + '.py')
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())
        # justify if there is pytest imported
        imports = []
        for n in tree.body:
            if isinstance(n, ast.Import):
                imports += [alias.name for alias in n.names]
            elif isinstance(n, ast.ImportFrom):
                imports += [n.module + '.' + alias.name for alias in n.names]
        if 'pytest' not in imports:
            continue

    # add @pytest.mark.order(786) to each test
    for func_def in tree.body:
        if isinstance(func_def, ast.FunctionDef) and func_def.name == test['@name']:
            # check if there is an declorator already
            order_dec_index = 0
            found = False
            for i in range(len(func_def.decorator_list)):
                if func_def.decorator_list[i].func.attr == 'order':
                    order_dec_index = i
                    found = True

            generated_order_dec = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='pytest', ctx=ast.Load()),
                    attr='mark.order',
                    ctx=ast.Load(),
                ),
                args=[ast.Num(i)],
                keywords=[],
            )

            if found:
                func_def.decorator_list[order_dec_index] = generated_order_dec
            else:
                func_def.decorator_list.append(generated_order_dec)

            with open(file_path, 'w') as f:
                f.write(ast.unparse(tree))
