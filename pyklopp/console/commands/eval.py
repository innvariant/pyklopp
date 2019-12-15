import json
import os
import random
import sys
import time
import uuid
import torch
import numpy as np
import ignite

from ignite.contrib.handlers import ProgressBar
from ignite.engine import create_supervised_evaluator
from cleo import Command

from pyklopp import __version__


def subpackage_import(name):
    components = name.split('.')

    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)

    return mod


class EvalCommand(Command):
    """
    Evaluates a pre-trained model on a given test data set.

    eval
        {model : Path to the pytorch model file}
        {testset : Function to retrieve the test set based on the assembled configuration}
        {--m|modules=* : Optional module file to load.}
        {--c|config=* : Configuration JSON string or file path.}
        {--s|save= : Path (including optional name) to save the configuration to, e.g. sub/path/config.json}
    """

    def handle(self):
        pass
