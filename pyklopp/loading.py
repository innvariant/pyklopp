import os
import sys


def subpackage_import(name: str):
    components = name.split('.')

    try:
        mod = __import__(components[0])
        for comp in components[1:]:
            mod = getattr(mod, comp)
    except AttributeError as e:
        raise AttributeError('Could not import <{name}>. System path info: "{syspath}"'.format(
            name=name, syspath=sys.path
        ), e)

    return mod


def add_local_path_to_system(fn_info=None):
    # Add current absolute path to system path to load local modules
    # If initialized a module previously from a local module, then it must be available in path later again
    add_path = os.path.abspath('.')
    if add_path not in sys.path:
        sys.path.append(add_path)

        if fn_info is not None:
            fn_info('Added "%s" to path.' % add_path)


def remove_local_path_from_system():
    remove_path = os.path.abspath('.')
    sys.path = [path for path in sys.path if path != remove_path]


def load_modules(module_args: list) -> list:
    loaded_modules = []

    if module_args is None:
        module_args = []

    for module_option in module_args:
        module_option = str(module_option)
        possible_module_file_name = module_option + '.py' if not module_option.endswith('.py') else module_option
        if os.path.exists(possible_module_file_name):
            module_file_name = possible_module_file_name
            module_name = module_file_name.replace('.py', '')

            try:
                loaded_modules.append(__import__('.' + module_name, fromlist=['']))
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError('Could not import "%s". Have you added "." to your system path?' % module_name, e)

    return loaded_modules
