from cleo.application import Application
from pyklopp import __version__

from .commands.eval import EvalCommand
from .commands.init import InitCommand
from .commands.train import TrainCommand

application = Application("pyklopp", __version__)
application.add(InitCommand())
application.add(TrainCommand())
application.add(EvalCommand())

if __name__ == "__main__":
    application.run()