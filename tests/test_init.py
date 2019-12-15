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


def test_init_empty_module():
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

    try:
        command_tester.execute(module_name + ' --save=' + save_path)
    except:
        # Clean up temporary module file
        os.remove(module_file_path)
        shutil.rmtree(os.path.dirname(save_path))

    os.remove(module_file_path)
    shutil.rmtree(os.path.dirname(save_path))


