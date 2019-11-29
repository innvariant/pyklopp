import pypaddle.sparse

from cleo import Command
from pyklopp.initializer import Norm001Initializer


class BuildCommand(Command):
    """
    Builds a model

    build
        {model : Name for one of the available (known) models.}
        {shape : Input tensor shape definition}
        {--i|init : Initializer to use}
        {--s|save : Path to save the model to}
    """

    def handle(self):
        model = self.argument('model')
        if model not in self.available_models.keys():
            raise ValueError('Unknown model ', model)

        init = self.available_initializers.keys()[0]
        if self.argument('init') in self.available_initializers.keys():
            init = self.argument('init')

        # Extract shape from arguments
        shape_description = self.argument('shape').strip().replace('(', '').replace(')', '')
        shape = tuple(int(d) for d in shape_description.split(',') if d)

        self.line('\tModel: %s' % str(model))
        self.line('\tShape: %s' % str(shape))
        self.line('\tInit: %s' % init)

    @property
    def available_initializers(self):  # type: () -> dict
        inits = [
            Norm001Initializer()
        ]
        return {str(i): i for i in inits}

    @property
    def available_models(self):  # type: () -> dict
        return {
            'MaskedDeepFFN': pypaddle.sparse.MaskedDeepFFN,
            'MaskedDeepDAN': pypaddle.sparse.MaskedDeepDAN
        }
