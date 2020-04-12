import pytest
import os
import shutil

from cleo import Application
from cleo import CommandTester

from pyklopp.console.commands.init import InitCommand


def test_value_error_on_unknown_module():
    application = Application()
    application.add(InitCommand())

    command = application.find('init')
    command_tester = CommandTester(command)

    unknown_module_name = 'some_module'
    with pytest.raises(ValueError, match=r".*%s.*" % unknown_module_name):
        command_tester.execute(unknown_module_name)


def test_value_error_on_invalid_config_with_single_quotes():
    application = Application()
    application.add(InitCommand())
    invalid_config = '{\'output_size\': 10}'
    module_name = 'foo'
    command = application.find('init')
    command_tester = CommandTester(command)

    with pytest.raises(ValueError, match=r".*{reason}.*".format(reason='double quotes')):
        command_tester.execute(module_name + ' --config "' + invalid_config + '"')


def test_value_error_on_unknown_model_getter():
    application = Application()
    application.add(InitCommand())

    module_name = 'foo'
    module_file_path = module_name + '.py'
    save_path = 'foo-config/model.py'
    content = '''
import torch

def init(**args):
    return torch.nn.Conv2d(10, 10, 10)

'''
    with open(module_file_path, 'a') as the_file:
        the_file.write(content)

    command = application.find('init')
    command_tester = CommandTester(command)

    with pytest.raises(ValueError, match=r".*{fn_name}.*".format(fn_name='get_model')):
        command_tester.execute(module_name + ' --save=' + save_path)

    os.remove(module_file_path)
    shutil.rmtree(os.path.dirname(save_path))


def test_success_init_simple_model():
    # Arrange
    ## set up application with command
    application = Application()
    application.add(InitCommand())

    # set up file path variables
    module_name = 'bar'
    module_file_path = module_name + '.py'
    save_path = 'bar-config/model.py'

    # clean up possible existing files
    if os.path.exists(module_file_path):
        os.remove(module_file_path)
    if os.path.exists(save_path):
        shutil.rmtree(os.path.dirname(save_path))

    # write model to file path from which we want to import from
    content = '''
import torch

def get_model(**args):
    return torch.nn.Conv2d(10, 10, 10)

'''
    with open(module_file_path, 'a') as the_file:
        the_file.write(content)

    # load command and build tester object to act on
    command = application.find('init')
    command_tester = CommandTester(command)

    # Act
    command_tester.execute(module_name + ' --save=' + save_path)

    # Cleanup
    os.remove(module_file_path)
    shutil.rmtree(os.path.dirname(save_path))



