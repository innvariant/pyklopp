import cleo
import pytest
import os

from unittest.mock import Mock, MagicMock
from pyfakefs.fake_filesystem import FakeFilesystem

from pyklopp.console.commands.init import InitCommand
from pyklopp.util import save_paths_obtain_and_check, load_modules, add_local_path, build_default_config


def test_save_paths_obtain_and_check(fs: FakeFilesystem):
    # `fs` is a plugin/fixture from pyfakefs
    # Arrange:
    command = Mock()
    command.option.return_value = 'hello/my_config.json'

    save_path_base, model_file_name = save_paths_obtain_and_check(command)

    command.option.assert_called()
    assert os.path.exists(save_path_base)
    assert not os.path.exists(os.path.join(save_path_base, model_file_name))
    assert len(save_path_base) > 0
    assert len(model_file_name) > 0


def test_save_paths_obtain_and_check_error_on_existing_file(fs: FakeFilesystem):
    command = Mock()
    path_existing_base_dir = 'existing/path/'
    os.makedirs(path_existing_base_dir)
    path_existing_config = os.path.join(path_existing_base_dir, 'my_config.json')
    fs.create_file(path_existing_config)
    command.option.return_value = path_existing_config

    with pytest.raises(ValueError, match=r".*{reason}.*".format(reason='already exists')):
        save_paths_obtain_and_check(command)


def test_load_modules():
    # Arrange
    local_module_name = 'my_module'
    local_module_file = local_module_name + '.py'
    definition_my_module = """class TestMyModule(object):
   def my_method(arg):
     return arg
"""
    with open(local_module_file, 'w') as file_handle:
        file_handle.write(definition_my_module)
    # Make sure the local path is added to enable local imports
    add_local_path()

    loaded_modules = load_modules([local_module_name])

    assert loaded_modules is not None
    assert len(loaded_modules) > 0

    # Cleanup
    os.remove(local_module_file)


def test_load_modules_error_non_existing_module_file():
    # Arrange
    non_existing_module_name = 'non_existing_module'
    # Make sure the local path is added to enable local imports
    add_local_path()

    with pytest.raises(ModuleNotFoundError, match=r".*{reason}.*".format(reason='Could not import')):
        load_modules([non_existing_module_name])


def test_load_modules_error_local_path_not_added():
    # Arrange
    non_existing_module_name = 'non_existing_module'
    non_existing_module_file = non_existing_module_name + '.py'
    definition_my_module = ''
    with open(non_existing_module_file, 'w') as file_handle:
        file_handle.write(definition_my_module)

    with pytest.raises(ModuleNotFoundError, match=r".*{reason}.*".format(reason='Could not import')):
        load_modules([non_existing_module_name])

    # Cleanup
    os.remove(non_existing_module_file)


def test_build_default_config():
    class TestCommand(cleo.Command):
        """
        Runs a function within a testing command environment.

        test
            {arg0 : This is an argument.}
            {--o|options=* : Possible options.}
            {--f|file= : Single option.}
        """
        def handle(self):
            build_default_config(self)

    application = cleo.Application()
    application.add(TestCommand())

    command = application.find('test')
    command_tester = cleo.CommandTester(command)

    command_tester.execute('val1 --o=opt1 --o=opt2 --f opt3')
