import pytest

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
