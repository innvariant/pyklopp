from cleo import Application as BaseApplication

from pyklopp import __version__

from .commands.eval import EvalCommand
from .commands.init import InitCommand
from .commands.train import TrainCommand


class Application(BaseApplication):
    def __init__(self):
        super(Application, self).__init__("pyklopp", __version__)

        for command in self.get_default_commands():
            self.add(command)

    def get_default_commands(self):  # type: () -> list
        commands = [InitCommand(), TrainCommand(), EvalCommand()]

        return commands
