import importlib, sys, os, pkg_resources, re

# method_id -> method
METHODS = {}

# dynamically export all included methods
def _find_builtin_methods(methods):
    d = os.path.dirname(__file__)
    pkg = '.'.join(__name__.split('.')[:-1])
    for mod_name in os.listdir(d):
        if os.path.isdir(os.path.join(d, mod_name)) and '__' not in mod_name:
            methods[mod_name] = f'{pkg}.{mod_name}.method'
    

def _find_external_methods(methods):
    regex = r"semb-method\[.+\]"
    for pkg in pkg_resources.working_set:
        matches = re.findall(regex, pkg.key)
        if len(matches) > 0:
            match = matches[0]
            methods[match.split('[')[1][:-1]] = match

_find_builtin_methods(METHODS)
_find_external_methods(METHODS)

def get_method_ids():
    global METHODS
    return list(METHODS.keys())

def load(method_id):
    global METHODS
    return importlib.import_module(METHODS[method_id]).Method
