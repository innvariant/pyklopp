import os

import pytest

from pyklopp.loading import add_local_path_to_system
from pyklopp.loading import load_modules
from pyklopp.loading import remove_local_path_from_system


def test_load_modules():
    # Arrange
    local_module_name = "my_module"
    local_module_file = local_module_name + ".py"
    definition_my_module = """class TestMyModule(object):
   def my_method(self, arg):
     return arg
"""
    with open(local_module_file, "w") as file_handle:
        file_handle.write(definition_my_module)
    # Make sure the local path is added to enable local imports
    add_local_path_to_system()

    loaded_modules = load_modules([local_module_name])

    assert loaded_modules is not None
    assert len(loaded_modules) > 0
    print(loaded_modules)

    test_arg = "5"
    obj = loaded_modules[0].TestMyModule()
    test_res = obj.my_method(test_arg)
    assert test_arg == test_res

    # Cleanup
    remove_local_path_from_system()
    os.remove(local_module_file)


def test_load_modules_subfolder():
    # Arrange
    folder = "submod"
    os.makedirs(folder)
    module_name = "my_module"
    module_full = folder + "." + module_name
    module_path = os.path.join(folder, module_name + ".py")
    definition_my_module = """class TestMyModule(object):
   def my_method(self, arg):
     return arg
"""
    with open(module_path, "w") as file_handle:
        file_handle.write(definition_my_module)
    # Make sure the local path is added to enable local imports
    add_local_path_to_system()

    loaded_modules = load_modules([module_full])

    assert loaded_modules is not None
    assert len(loaded_modules) > 0
    print(loaded_modules)

    test_arg = "5"
    obj = loaded_modules[0].TestMyModule()
    test_res = obj.my_method(test_arg)
    assert test_arg == test_res

    # Cleanup
    remove_local_path_from_system()
    os.remove(module_path)
    os.removedirs(folder)


def test_load_modules_error_non_existing_module_file():
    # Arrange
    non_existing_module_name = "non_existing_module"
    # Make sure the local path is added to enable local imports
    add_local_path_to_system()

    with pytest.raises(ModuleNotFoundError):
        load_modules([non_existing_module_name])

    # Cleanup
    remove_local_path_from_system()


def test_load_modules_error_local_path_not_added():
    # Arrange
    empty_module_name = "empty_module"
    empty_module_file = empty_module_name + ".py"
    definition_my_module = ""
    with open(empty_module_file, "w") as file_handle:
        file_handle.write(definition_my_module)
    # Make sure current path is not in sys.path
    remove_local_path_from_system()

    with pytest.raises(
        ModuleNotFoundError, match=r".*{reason}.*".format(reason="Could not import")
    ):
        load_modules([empty_module_name])

    # Cleanup
    os.remove(empty_module_file)
