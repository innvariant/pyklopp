import json
import os
import uuid

from cleo import Command


class PackCommand(Command):
    """
    Foo

    pack
        {name : Name}
        {--m|module=* : Optional module file to load.}
        {--s|save= : Path }
    """

    def handle(self):
        modules_option = self.option("module")

        print(modules_option)
